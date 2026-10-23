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
