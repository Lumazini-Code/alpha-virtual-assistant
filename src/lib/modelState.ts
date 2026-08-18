import { DEFAULT_MMPROJ_USED, DEFAULT_MODEL } from "./constants.js";

interface ChosenModel {
  model: string;
  mmprojUsed: boolean;
}

// Estado simples em memória, escrito uma vez pelo ModelSelector (ou pelas
// env vars AVA_MODEL/AVA_MMPROJ_USED como default) e lido por
// ensureProcessesStarted() a cada pergunta.
let current: ChosenModel = {
  model: DEFAULT_MODEL,
  mmprojUsed: DEFAULT_MMPROJ_USED,
};

export function getChosenModel(): ChosenModel {
  return current;
}

export function setChosenModel(model: string, mmprojUsed: boolean): void {
  current = { model, mmprojUsed };
}
