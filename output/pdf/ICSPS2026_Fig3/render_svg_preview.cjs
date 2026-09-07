// Optional independent SVG rendering check. Requires sharp (tested: 0.35.4).
// Set SHARP_MODULE to an absolute sharp package path when needed.
const path = require('path');
const sharp = require(process.env.SHARP_MODULE || 'sharp');
const input = process.argv[2] || path.join(__dirname, 'fig3_representation_ams.svg');
const output = process.argv[3] || path.join(__dirname, 'fig3_svg_render_check.png');
sharp(input, { density: 72 }).resize({ width: Math.round(252 / 72.27 * 150) })
  .png().toFile(output)
  .then(info => process.stdout.write(JSON.stringify({ sharp: sharp.versions.sharp, ...info }, null, 2)))
  .catch(err => { process.stderr.write(String(err)); process.exitCode = 1; });
