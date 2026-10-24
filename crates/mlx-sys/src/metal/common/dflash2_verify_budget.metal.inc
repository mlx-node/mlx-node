// B1 specialization of Splash draft_verify_rows (Apache-2.0).
// parameters: slope, 7 intercepts, 9 costs, every_row. Shapes stay fixed.
R"(
  {
    uint rows = 8;
    if (parameters[17] == 0.0f) {
      rows = 1;
      float survival = 1.0f;
      float expected = 1.0f;
      float best = expected / parameters[9];
      for (uint position = 0; position < 7; ++position) {
        const float q = clamp(confidence[position], 1.0e-6f, 1.0f - 1.0e-6f);
        const float z = parameters[0] * log(q / (1.0f - q)) + parameters[1 + position];
        survival *= 1.0f / (1.0f + exp(-z));
        expected += survival;
        const float value = expected / parameters[10 + position];
        if (value > best) { best = value; rows = position + 2; }
      }
    }
    budget[0] = int(rows);
  }
)"
