// Reescreve cfonts/lib/GetFont.js para resolver as fontes com require ESTÁTICO.
//
// Por que: o GetFont original usa require(path.normalize(`../fonts/${font}.json`))
// — path computado, que o bun build --compile não consegue seguir. Num binário
// compilado isso quebra em runtime ("Font file could not be found").
// Com requires estáticos o bundler embute os JSONs direto no binário.
//
// Idempotente e roda antes de cada build (npm ci recria o node_modules
// original; este script re-aplica o patch automaticamente).

import { readdirSync, writeFileSync, readFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const FONTS_DIR = join(ROOT, "node_modules", "cfonts", "fonts");
const TARGET = join(ROOT, "node_modules", "cfonts", "lib", "GetFont.js");

const marker = "[alpha-ai static fonts]";
if (readFileSync(TARGET, "utf8").includes(marker)) {
  process.exit(0);
}

if (!existsSync(FONTS_DIR)) {
  console.error("patch-cfonts: node_modules/cfonts/fonts não encontrado — rode bun install primeiro.");
  process.exit(1);
}

const fonts = readdirSync(FONTS_DIR)
  .filter((f) => f.endsWith(".json"))
  .map((f) => f.replace(/\.json$/, ""));

const requires = fonts.map((f) => `  '${f}': require('../fonts/${f}.json'),`).join("\n");

const patched = `// ${marker}
// Gerado por scripts/patch-cfonts.mjs — não editar.
'use strict';

const FONTFILES = {
${requires}
};

const GetFont = (font) => {
  if (Object.prototype.hasOwnProperty.call(FONTFILES, font)) {
    return FONTFILES[font];
  }
  return false;
};

module.exports = exports = { GetFont };
`;

writeFileSync(TARGET, patched);
console.log(`patch-cfonts: ${fonts.length} fontes embutidas estaticamente (${fonts.join(", ")})`);
