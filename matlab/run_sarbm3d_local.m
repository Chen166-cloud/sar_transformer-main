function result = run_sarbm3d_local(package_root, output_root, input_path, looks, input_domain, mat_field, demo_source)
%RUN_SARBM3D_LOCAL Run the authors' SAR-BM3D v1.0 on a local image or demo.
% result = run_sarbm3d_local(package_root, output_root, input_path, L, ...
%                           input_domain, mat_field, demo_source)
% Empty input_path runs the authors' 256-by-256 Napoli demo by default.
% demo_source='synthetic' instead creates a deterministic 128-by-128 demo
% with multiplicative Gamma(L, 1/L) speckle. L defaults to 1. input_domain is
% 'intensity' (default) or 'amplitude'. MAT input uses the exact top-level
% field mat_field (default 'noisy'); grayscale raster and MAT numeric values
% retain their original scale and are converted to double precision.
%
% The original algorithm consumes and returns amplitude. Scientific arrays
% saved in result.mat are never clipped or independently rescaled. PNG files
% use one common intensity display range and are for visualization only.
% This wrapper does not contain or replace the authors' algorithm.

    if nargin < 2
        error('SARBM3D:Arguments', 'package_root and output_root are required.');
    end
    if nargin < 3 || isempty(input_path), input_path = ''; end
    if nargin < 4 || isempty(looks), looks = 1; end
    if nargin < 5 || isempty(input_domain), input_domain = 'intensity'; end
    if nargin < 6 || isempty(mat_field), mat_field = 'noisy'; end
    if nargin < 7 || isempty(demo_source), demo_source = 'official'; end
    package_root = text_scalar(package_root, 'package_root');
    output_root = text_scalar(output_root, 'output_root');
    input_path = text_scalar(input_path, 'input_path');
    input_domain = lower(text_scalar(input_domain, 'input_domain'));
    mat_field = text_scalar(mat_field, 'mat_field');
    demo_source = lower(text_scalar(demo_source, 'demo_source'));
    if isempty(package_root) || exist(package_root, 'dir') ~= 7
        error('SARBM3D:Package', 'Package directory does not exist: %s', package_root);
    end
    if isempty(output_root)
        error('SARBM3D:Output', 'output_root must name an output directory.');
    end
    validateattributes(looks, {'numeric'}, {'real', 'finite', 'scalar', '>=', 1}, mfilename, 'looks');
    looks = double(looks);
    if ~any(strcmp(input_domain, {'intensity', 'amplitude'}))
        error('SARBM3D:Domain', 'input_domain must be intensity or amplitude.');
    end
    if ~isvarname(mat_field)
        error('SARBM3D:MATField', 'mat_field must be an exact top-level MAT variable name.');
    end
    if ~any(strcmp(demo_source, {'official', 'synthetic'}))
        error('SARBM3D:Demo', 'demo_source must be official or synthetic.');
    end

    artifact_names = {'result.mat', 'input.png', 'denoised.png', 'comparison.png', 'summary.json'};
    if exist(output_root, 'file') ~= 0 && exist(output_root, 'dir') ~= 7
        error('SARBM3D:Output', 'Output path is already a file: %s', output_root);
    end
    for k = 1:numel(artifact_names)
        destination = fullfile(output_root, artifact_names{k});
        if exist(destination, 'file') ~= 0
            error('SARBM3D:OutputExists', 'Refusing to overwrite existing output: %s', destination);
        end
    end

    original_matlab_path = path;
    original_system_path = getenv('PATH');
    environment_cleanup = onCleanup(@() restore_environment(original_matlab_path, original_system_path));
    package_paths = genpath(package_root);
    addpath(package_paths);
    % Win64 MEX binaries depend on DLLs shipped in the archive. Changes apply
    % only to this MATLAB process and are restored when this function exits.
    package_directories = strsplit(package_paths, pathsep);
    dll_directories = {};
    for k = 1:numel(package_directories)
        candidate = package_directories{k};
        if ~isempty(candidate) && ~isempty(dir(fullfile(candidate, '*.dll')))
            dll_directories{end + 1} = candidate; %#ok<AGROW>
        end
    end
    if ~isempty(dll_directories)
        setenv('PATH', [strjoin(dll_directories, pathsep), pathsep, original_system_path]);
    end
    algorithm_path = which('SARBM3D_v10');
    if isempty(algorithm_path)
        error('SARBM3D:Package', 'SARBM3D_v10 was not found under %s.', package_root);
    end

    seed = 20260908;
    demo_mode = isempty(input_path);
    synthetic_demo = demo_mode && strcmp(demo_source, 'synthetic');
    official_demo = demo_mode && strcmp(demo_source, 'official');
    clean_intensity = [];
    metric_range = 1;
    if synthetic_demo
        original_rng = rng;
        rng_cleanup = onCleanup(@() rng(original_rng));
        rng(seed, 'twister');
        clean_intensity = make_demo_intensity(128);
        noisy_intensity = clean_intensity .* (randg(looks, size(clean_intensity)) ./ looks);
        noisy_amplitude = sqrt(noisy_intensity);
        effective_input_domain = 'intensity';
        input_conversion = 'synthetic_clean_intensity_times_unclipped_gamma_speckle';
    elseif official_demo
        if looks ~= 1
            error('SARBM3D:DemoLooks', 'The official napoli_noisy.raw demo has L=1. Use synthetic for another L.');
        end
        algorithm_directory = fileparts(algorithm_path);
        clean_amplitude = read_official_amplitude(fullfile(algorithm_directory, 'napoli.raw'));
        original_noisy_amplitude = read_official_amplitude(fullfile(algorithm_directory, 'napoli_noisy.raw'));
        clean_intensity = clean_amplitude .^ 2;
        noisy_intensity = original_noisy_amplitude .^ 2;
        noisy_amplitude = original_noisy_amplitude;
        metric_range = 255 ^ 2;
        effective_input_domain = 'amplitude';
        input_conversion = 'official_raw_float32_little_endian_amplitude_values_preserved';
    else
        [raw_input, input_conversion] = read_input(input_path, mat_field);
        validate_image(raw_input, 'Input image');
        if strcmp(input_domain, 'amplitude')
            noisy_intensity = raw_input .^ 2;
            noisy_amplitude = raw_input;
        else
            noisy_intensity = raw_input;
            noisy_amplitude = sqrt(noisy_intensity);
        end
        effective_input_domain = input_domain;
    end
    has_reference = ~isempty(clean_intensity);
    validate_image(noisy_intensity, 'Input intensity');
    if ~any(noisy_intensity(:) > 0)
        error('SARBM3D:Input', 'Input must contain at least one positive pixel.');
    end

    if exist(output_root, 'dir') ~= 7
        [ok, message] = mkdir(output_root);
        if ~ok, error('SARBM3D:Output', 'Cannot create output directory: %s', message); end
    end
    fprintf('SAR-BM3D v1.0 | %d x %d | L = %.8g | input domain: %s\n', ...
        size(noisy_intensity, 1), size(noisy_intensity, 2), looks, effective_input_domain);
    fprintf('Authors'' implementation: %s\n', algorithm_path);
    inference_clock = tic;
    denoised_amplitude = SARBM3D_v10(noisy_amplitude, looks);
    inference_seconds = toc(inference_clock);
    if ~isnumeric(denoised_amplitude) || ~isreal(denoised_amplitude) || ...
            ~isequal(size(denoised_amplitude), size(noisy_amplitude)) || ...
            any(~isfinite(denoised_amplitude(:)))
        error('SARBM3D:Prediction', 'SARBM3D_v10 returned an invalid amplitude image.');
    end
    denoised_amplitude = double(denoised_amplitude);
    denoised_intensity = denoised_amplitude .^ 2;
    validate_image(denoised_intensity, 'Output intensity');

    if has_reference
        display_upper = metric_range;
    else
        display_values = sort([noisy_intensity(:); denoised_intensity(:)]);
        display_upper = display_values(max(1, ceil(0.99 * numel(display_values))));
        if display_upper <= 0, display_upper = max(display_values); end
    end
    input_display = intensity_preview(noisy_intensity, display_upper);
    denoised_display = intensity_preview(denoised_intensity, display_upper);
    separator = repmat(uint8(255), size(input_display, 1), 6);
    if has_reference
        comparison = [intensity_preview(clean_intensity, display_upper), separator, ...
            input_display, separator, denoised_display];
        panel_order = {'clean_intensity', 'noisy_intensity', 'denoised_intensity'};
    else
        comparison = [input_display, separator, denoised_display];
        panel_order = {'noisy_intensity', 'denoised_intensity'};
    end

    summary = struct;
    summary.schema_version = 1;
    summary.algorithm = 'SAR-BM3D v1.0 (authors'' SARBM3D_v10)';
    summary.algorithm_path = algorithm_path;
    summary.package_root = package_root;
    summary.matlab_version = version;
    summary.matlab_release = version('-release');
    summary.architecture = computer('arch');
    summary.mex_extension = mexext;
    summary.synthetic_demo = synthetic_demo;
    summary.official_demo = official_demo;
    summary.has_reference = has_reference;
    if demo_mode, summary.demo_source = demo_source; else, summary.demo_source = 'none'; end
    summary.input_path = input_path;
    summary.mat_field = mat_field;
    summary.input_domain = effective_input_domain;
    summary.input_conversion = input_conversion;
    summary.algorithm_domain = 'amplitude_sqrt_linear_intensity';
    summary.output_domain = 'linear_intensity';
    if synthetic_demo, summary.rng_seed = seed; else, summary.rng_seed = []; end
    summary.looks = looks;
    summary.image_size = size(noisy_intensity);
    summary.inference_seconds = inference_seconds;
    summary.input_intensity_min = min(noisy_intensity(:));
    summary.input_intensity_max = max(noisy_intensity(:));
    summary.denoised_intensity_min = min(denoised_intensity(:));
    summary.denoised_intensity_max = max(denoised_intensity(:));
    summary.denoised_amplitude_min = min(denoised_amplitude(:));
    summary.denoised_amplitude_max = max(denoised_amplitude(:));
    summary.scientific_arrays_clipped = false;
    summary.display_intensity_range = [0, display_upper];
    summary.png_is_display_only = true;
    summary.comparison_panel_order = panel_order;
    summary.metrics = struct('available', has_reference);
    if has_reference
        summary.metrics.domain = 'unclipped_linear_intensity';
        summary.metrics.data_range = metric_range;
        summary.metrics.noisy_mse = mean((noisy_intensity(:) - clean_intensity(:)) .^ 2);
        summary.metrics.denoised_mse = mean((denoised_intensity(:) - clean_intensity(:)) .^ 2);
        summary.metrics.noisy_psnr_db = 10 * log10(metric_range ^ 2 / summary.metrics.noisy_mse);
        summary.metrics.denoised_psnr_db = 10 * log10(metric_range ^ 2 / summary.metrics.denoised_mse);
        summary.metrics.ssim_available = false;
        if exist('ssim', 'file') == 2
            try
                noisy_ssim = ssim(noisy_intensity, clean_intensity, 'DynamicRange', metric_range);
                denoised_ssim = ssim(denoised_intensity, clean_intensity, 'DynamicRange', metric_range);
                summary.metrics.noisy_ssim = noisy_ssim;
                summary.metrics.denoised_ssim = denoised_ssim;
                summary.metrics.ssim_available = true;
            catch ssim_error
                summary.metrics.ssim_unavailable_reason = ssim_error.message;
            end
        end
        if official_demo
            summary.metrics.official_amplitude_data_range = 255;
            summary.metrics.official_amplitude_noisy_psnr_db = 10 * log10(255 ^ 2 / ...
                mean((noisy_amplitude(:) - clean_amplitude(:)) .^ 2));
            summary.metrics.official_amplitude_denoised_psnr_db = 10 * log10(255 ^ 2 / ...
                mean((denoised_amplitude(:) - clean_amplitude(:)) .^ 2));
            fprintf('Official amplitude PSNR: noisy %.3f dB; SAR-BM3D %.3f dB (data range 255).\n', ...
                summary.metrics.official_amplitude_noisy_psnr_db, ...
                summary.metrics.official_amplitude_denoised_psnr_db);
        end
        fprintf('Intensity PSNR: noisy %.3f dB; SAR-BM3D %.3f dB (data range %.8g).\n', ...
            summary.metrics.noisy_psnr_db, summary.metrics.denoised_psnr_db, metric_range);
    end

    result = struct('noisy_intensity', noisy_intensity, ...
        'noisy_amplitude', noisy_amplitude, 'denoised_intensity', denoised_intensity, ...
        'denoised_amplitude', denoised_amplitude, 'clean_intensity', clean_intensity, ...
        'summary', summary);
    save(fullfile(output_root, 'result.mat'), '-struct', 'result', '-v7');
    imwrite(input_display, fullfile(output_root, 'input.png'));
    imwrite(denoised_display, fullfile(output_root, 'denoised.png'));
    imwrite(comparison, fullfile(output_root, 'comparison.png'));
    json_path = fullfile(output_root, 'summary.json');
    [json_file, message] = fopen(json_path, 'w', 'n', 'UTF-8');
    if json_file < 0, error('SARBM3D:Output', 'Cannot write summary: %s', message); end
    json_cleanup = onCleanup(@() fclose(json_file));
    fprintf(json_file, '%s\n', jsonencode(summary));
    fprintf('Finished in %.3f seconds. Results: %s\n', inference_seconds, output_root);
end

function amplitude = read_official_amplitude(raw_path)
    info = dir(raw_path);
    if isempty(info) || info.bytes ~= 256 * 256 * 4
        error('SARBM3D:Demo', 'Expected the original 256 x 256 float32 demo file: %s', raw_path);
    end
    [raw_file, message] = fopen(raw_path, 'r', 'ieee-le');
    if raw_file < 0, error('SARBM3D:Demo', 'Cannot open demo file: %s', message); end
    raw_cleanup = onCleanup(@() fclose(raw_file));
    amplitude = fread(raw_file, [256, 256], 'single=>double')';
    validate_image(amplitude, 'Official demo amplitude');
end

function value = text_scalar(value, name)
    if isstring(value) && isscalar(value), value = char(value); end
    if ~ischar(value) || (~isempty(value) && ~isrow(value))
        error('SARBM3D:Arguments', '%s must be a character row or scalar string.', name);
    end
end

function validate_image(value, name)
    if ~isnumeric(value) || ~isreal(value) || ~ismatrix(value) || isempty(value) || ...
            any(~isfinite(value(:))) || any(value(:) < 0)
        error('SARBM3D:Input', '%s must be a finite, real, nonnegative 2-D numeric array.', name);
    end
    if any(size(value) < 64)
        error('SARBM3D:InputSize', '%s must be at least 64 x 64 pixels.', name);
    end
end

function [value, conversion] = read_input(input_path, mat_field)
    if exist(input_path, 'file') ~= 2
        error('SARBM3D:Input', 'Input file does not exist: %s', input_path);
    end
    [~, ~, extension] = fileparts(input_path);
    if strcmpi(extension, '.mat')
        payload = load(input_path, mat_field);
        if ~isfield(payload, mat_field)
            error('SARBM3D:MATField', 'MAT file does not contain the exact field "%s".', mat_field);
        end
        if ~isnumeric(payload.(mat_field)) && ~islogical(payload.(mat_field))
            error('SARBM3D:Input', 'MAT field must be a numeric image array.');
        end
        value = full(double(payload.(mat_field)));
        conversion = 'MAT_numeric_values_preserved';
        return;
    end
    if strcmpi(extension, '.npy')
        error('SARBM3D:InputFormat', 'NPY input is unsupported; export the array to a MAT file.');
    end
    [pixels, map] = imread(input_path);
    if ~isempty(map)
        pixels = ind2rgb(pixels, map);
        conversion = 'indexed_raster_palette_to_RGB';
    elseif isinteger(pixels)
        pixels = double(pixels);
        conversion = 'integer_raster_numeric_values_preserved';
    else
        pixels = double(pixels);
        conversion = 'floating_or_logical_raster_values_preserved';
    end
    if ndims(pixels) == 3 && size(pixels, 3) == 3
        value = 0.2989360213 * pixels(:, :, 1) + ...
            0.5870430745 * pixels(:, :, 2) + 0.1140209043 * pixels(:, :, 3);
        conversion = [conversion, ';RGB_to_luminance'];
    elseif ismatrix(pixels)
        value = pixels;
    else
        error('SARBM3D:Input', 'Raster must be grayscale, indexed, or RGB.');
    end
end

function clean = make_demo_intensity(side_length)
    [x, y] = meshgrid(linspace(-1, 1, side_length));
    clean = 0.12 + 0.08 * (x + 1) / 2 + 0.04 * (y + 1) / 2;
    clean(x > -0.82 & x < -0.16 & y > -0.75 & y < -0.15) = 0.60;
    clean((x - 0.43) .^ 2 + (y + 0.43) .^ 2 < 0.24 ^ 2) = 0.90;
    clean((x + 0.43) .^ 2 + (y - 0.45) .^ 2 < 0.28 ^ 2) = 0.055;
    texture = x > 0.04 & x < 0.88 & y > 0.18 & y < 0.80;
    clean(texture) = 0.38 + 0.10 * sin(35 * x(texture)) .* cos(25 * y(texture));
    clean(abs(y - 0.32 * x - 0.03) < 0.022) = 0.76;
    clean(abs(x - 0.76) < 0.015 & y < 0) = 1;
end

function preview = intensity_preview(intensity, upper)
    preview = uint8(round(255 * min(1, max(0, intensity ./ upper))));
end

function restore_environment(matlab_path, system_path)
    path(matlab_path);
    setenv('PATH', system_path);
end
