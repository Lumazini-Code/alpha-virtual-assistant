// Alpha AI (AVA) — logo ASCII da AVA
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
import React from "react";
import { Box, Text } from "ink";
import BigText from "ink-big-text";
import { COLORS, TAGLINE } from "../lib/constants.js";

export function Logo() {
  return (
    <Box flexDirection="column" alignItems="center" marginBottom={1}>
      <BigText text="ALPHA AI" font="block" colors={[COLORS.blue]} />
      <Text color={COLORS.gray}>{TAGLINE}</Text>
    </Box>
  );
}
