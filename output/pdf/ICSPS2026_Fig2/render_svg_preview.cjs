// Optional independent SVG rendering check, requires the sharp Node package.
// node output/pdf/ICSPS2026_Fig2/render_svg_preview.cjs
// Set SHARP_MODULE to an absolute package path if sharp is not installed locally.
const path = require('path');
const sharp = require(process.env.SHARP_MODULE || 'sharp');
const input = process.argv[2] || path.join(__dirname, 'fig2_fdr_block.svg');
const output = process.argv[3] || path.join(__dirname, 'fig2_svg_render_check.png');
const width = Math.round(252 / 72.27 * 150);
sharp(input, { density: 72 }).resize({ width }).png().toFile(output)
  .then(info => process.stdout.write(JSON.stringify({ sharp: sharp.versions, ...info }, null, 2)))
  .catch(err => { process.stderr.write(String(err)); process.exitCode = 1; });
