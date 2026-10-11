/**
 * Link clicável de terminal via OSC 8 (suportado por GNOME Terminal,
 * Konsole, Kitty, Alacritty, iTerm2, Windows Terminal…). Em terminais
 * sem suporte o texto aparece plano — inofensivo.
 *
 * IMPORTANTE: o terminador precisa ser ST ("\x1b\\") e NÃO BEL ("\x07") —
 * o Ink mede largura com string-width/ansi-regex, que só desconta a
 * sequência quando ela termina com ST (verificado: ST → largura correta,
 * BEL → largura inclui os escapes e quebra o layout).
 */
export function terminalLink(text: string, url: string): string {
  return `\x1b]8;;${url}\x1b\\${text}\x1b]8;;\x1b\\`;
}
