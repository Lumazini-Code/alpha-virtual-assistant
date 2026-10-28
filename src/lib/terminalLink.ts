// Alpha AI (AVA) — links clicáveis de terminal via OSC 8
// Copyright (C) 2026  Felipe lumazini
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
