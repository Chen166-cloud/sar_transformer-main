function run_icsps2026_sarbm3d(package_root, job_csv, pair_root, output_root)
%RUN_ICSPS2026_SARBM3D Run GRIP-UNINA SAR-BM3D v1.0 on frozen pair jobs.
%
% This wrapper contains no SAR-BM3D source. Download the authors' package,
% accept its nonprofit-only license, and pass its directory as package_root.
% Each completed pair is saved independently, so an interrupted invocation
% can be restarted without overwriting finished predictions.

    if exist(package_root, 'dir') ~= 7
        error('SAR-BM3D package directory does not exist: %s', package_root);
    end
    % The official archive may contain a versioned top-level directory, so
    % add its subtree while keeping the external package separate from ours.
    addpath(genpath(package_root));
    if exist('SARBM3D_v10', 'file') == 0
        error('SARBM3D_v10 is not on the MATLAB path after adding %s', package_root);
    end
    if exist(job_csv, 'file') ~= 2
        error('Job CSV does not exist: %s', job_csv);
    end
    if exist(pair_root, 'dir') ~= 7
        error('Pair root does not exist: %s', pair_root);
    end
    if exist(output_root, 'dir') ~= 7
        mkdir(output_root);
    end
    prediction_dir = fullfile(output_root, 'predictions');
    if exist(prediction_dir, 'dir') ~= 7
        mkdir(prediction_dir);
    end

    jobs = readtable(job_csv, 'TextType', 'string');
    required = {'pair_id', 'mat_path', 'noisy_field', 'L', 'output_relative_path'};
    if ~all(ismember(required, jobs.Properties.VariableNames))
        error('Job CSV is missing required columns.');
    end
    diary_path = fullfile(output_root, 'matlab_stdout.log');
    diary(diary_path);
    cleanup_diary = onCleanup(@() diary('off'));
    fprintf('SAR-BM3D jobs: %d\n', height(jobs));
    help SARBM3D_v10;

    for index = 1:height(jobs)
        pair_id = char(jobs.pair_id(index));
        input_path = fullfile(pair_root, char(jobs.mat_path(index)));
        output_path = fullfile(output_root, char(jobs.output_relative_path(index)));
        if exist(output_path, 'file') == 2
            fprintf('[%d/%d] skip existing %s\n', index, height(jobs), pair_id);
            continue;
        end
        payload = load(input_path, char(jobs.noisy_field(index)));
        noisy_intensity = double(payload.(char(jobs.noisy_field(index))));
        looks = double(jobs.L(index));
        if ndims(noisy_intensity) ~= 2 || any(~isfinite(noisy_intensity(:))) || ...
                min(noisy_intensity(:)) < 0 || max(noisy_intensity(:)) > 1
            error('Invalid normalized intensity input for pair %s', pair_id);
        end
        % The authors' SARBM3D_v10 interface explicitly consumes and returns
        % square-root intensity.  Our frozen protocol and all metrics use
        % linear intensity, so the two domain conversions are mandatory.
        noisy_amplitude = sqrt(noisy_intensity);
        started = tic;
        prediction_amplitude = SARBM3D_v10(noisy_amplitude, looks);
        inference_seconds = toc(started);
        if ~isequal(size(prediction_amplitude), size(noisy_amplitude)) || ...
                any(~isfinite(prediction_amplitude(:)))
            error('Invalid SAR-BM3D prediction for pair %s', pair_id);
        end
        prediction = prediction_amplitude .^ 2;
        if any(~isfinite(prediction(:)))
            error('Invalid squared intensity prediction for pair %s', pair_id);
        end
        prediction_amplitude_min = min(prediction_amplitude(:));
        prediction_amplitude_max = max(prediction_amplitude(:));
        input_domain = 'sqrt_linear_normalized_intensity';
        output_domain = 'linear_normalized_intensity_after_squaring';
        domain_conversion = 'sqrt_before_SARBM3D_v10;square_after';
        temporary_path = [output_path '.tmp.mat'];
        save(temporary_path, 'prediction', 'pair_id', 'looks', ...
            'inference_seconds', 'prediction_amplitude_min', ...
            'prediction_amplitude_max', 'input_domain', 'output_domain', ...
            'domain_conversion', '-v7');
        movefile(temporary_path, output_path, 'f');
        fprintf('[%d/%d] %s %.3f s\n', index, height(jobs), pair_id, inference_seconds);
    end
    clear cleanup_diary;
end
