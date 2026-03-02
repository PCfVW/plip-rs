(* Recurrent feedback experiment figures for Llama 3.2 1B *)
(* Source: outputs/recurrent_block_rhyme.json *)
(* Branch: anacrousis *)

(* ============================================================ *)
(* DATA                                                          *)
(* ============================================================ *)

(* --- Figure 1: Strength-response curve for L14-15 --- *)
(* Format: {strength, rhymes/15} *)

prefillOnly = {
  {0, 10},    (* baseline *)
  {1, 9},     (* unembed_14-15_s=1.0 *)
  {2, 10},    (* unembed_14-15_s=2.0 *)
  {5, 9},     (* unembed_14-15_s=5.0 *)
  {10, 6},    (* unembed_14-15_s=10.0 *)
  {20, 5}     (* unembed_14-15_s=20.0 *)
};

sustained = {
  {0, 10},    (* baseline *)
  {0.5, 9},   (* sustained_14-15_s=0.5 *)
  {1, 11},    (* sustained_14-15_s=1.0 — NEW BEST *)
  {2, 7}      (* sustained_14-15_s=2.0 *)
};

(* --- Figure 2: Per-couplet success grid --- *)
(* Format: {id, target, baseline, prefill s=1.0, sustained s=1.0, prefill s=2.0} *)
(* 1 = rhyme (Y), 0 = no rhyme (X) *)

coupletGrid = {
  (* id   target     base  pf1.0  sus1.0  pf2.0 *)
  { 1,   "light",     0,     0,     1,      0},   (* CONVERTED by sustained *)
  { 2,   "play",      1,     1,     1,      1},
  { 3,   "sound",     1,     1,     1,      1},
  { 4,   "rain",      0,     0,     0,      0},   (* Resistant failure *)
  { 5,   "time",      0,     0,     0,      0},   (* Resistant failure *)
  { 6,   "air",       1,     1,     1,      1},
  { 7,   "gold",      0,     0,     0,      0},   (* Resistant failure *)
  { 8,   "fire",      1,     1,     1,      1},
  { 9,   "stone",     1,     1,     1,      1},
  {10,   "dream",     1,     1,     1,      1},
  {11,   "strange",   1,     1,     1,      1},
  {12,   "love",      1,     1,     1,      1},
  {13,   "truth",     0,     0,     0,      0},   (* Resistant failure *)
  {14,   "world",     1,     0,     1,      1},   (* Prefill s=1.0 REGRESSES; sustained PRESERVES *)
  {15,   "earth",     1,     1,     1,      1}
};

coupletLabels = {"Baseline", "Prefill s=1.0", "Sustained s=1.0", "Prefill s=2.0"};

(* Generated words for annotation (what the model actually produced) *)
generatedWords = {
  (* id   baseline       pf1.0          sus1.0         pf2.0 *)
  { 1,   "silvered",    "silvered",    "light",       "silvered"},
  { 2,   "play",        "play",        "play",        "play"},
  { 3,   "sound",       "sound",       "sound",       "sound"},
  { 4,   "of",          "of",          "sky",         "of"},
  { 5,   "that",        "that",        "that",        "that"},
  { 6,   "air",         "air",         "air",         "air"},
  { 7,   "setting",     "setting",     "setting",     "setting"},
  { 8,   "fire",        "fire",        "fire",        "fire"},
  { 9,   "stone",       "stone",       "stone",       "stone"},
  {10,   "dream",       "dream",       "dream",       "dream"},
  {11,   "strange",     "strange",     "strange",     "strange"},
  {12,   "love",        "love",        "love",        "love"},
  {13,   "girl",        "girl",        "girl",        "girl"},
  {14,   "world",       "man",         "world",       "world"},
  {15,   "earth",       "earth",       "earth",       "earth"}
};

(* ============================================================ *)
(* FIGURE 1: Strength-Response Curve                             *)
(* X-axis: Feedback strength                                     *)
(* Y-axis: Rhyme success (out of 15)                             *)
(* Two series: prefill-only (blue) and sustained (red/orange)    *)
(* Horizontal dashed line at baseline = 10                       *)
(* ============================================================ *)

fig1 = Show[
  ListLinePlot[
    {prefillOnly, sustained},
    PlotStyle -> {
      {Thick, RGBColor[0.2, 0.4, 0.8]},       (* Prefill: blue *)
      {Thick, RGBColor[0.85, 0.33, 0.1]}       (* Sustained: orange-red *)
    },
    PlotMarkers -> {
      {"\[FilledCircle]", 10},
      {"\[FilledSquare]", 10}
    },
    PlotLegends -> Placed[
      LineLegend[
        {RGBColor[0.2, 0.4, 0.8], RGBColor[0.85, 0.33, 0.1]},
        {Style["Prefill-only", 12], Style["Sustained", 12]},
        LegendMarkers -> {{"\[FilledCircle]", 10}, {"\[FilledSquare]", 10}}
      ],
      {0.75, 0.80}
    ],
    Frame -> True,
    FrameLabel -> {
      Style["Feedback strength", 14],
      Style["Rhyme success (/15)", 14]
    },
    PlotLabel -> Style["Recurrent Feedback on L14\[Dash]15 (Llama 3.2 1B)", 16, Bold],
    PlotRange -> {{-0.5, 21}, {-0.5, 15.5}},
    GridLines -> {None, {5, 10, 15}},
    GridLinesStyle -> Directive[GrayLevel[0.85]],
    ImageSize -> 700,
    AspectRatio -> 0.5,
    PlotRangePadding -> {{Scaled[0.02], Scaled[0.02]}, {0.3, 0.3}},
    FrameTicks -> {
      {{0, 0.5, 1, 2, 5, 10, 20}, Automatic},
      {Range[0, 15], Automatic}
    }
  ],
  (* Baseline reference line *)
  Graphics[{
    Dashed, GrayLevel[0.5], AbsoluteThickness[1.5],
    Line[{{-0.5, 10}, {21, 10}}],
    Text[
      Style["baseline = 10/15", 11, Italic, GrayLevel[0.4]],
      {14, 10.4}, {-1, -1}
    ]
  }],
  (* Annotate the peak: sustained s=1.0 = 11/15 *)
  Graphics[{
    RGBColor[0.85, 0.33, 0.1],
    Text[
      Style["11/15", 13, Bold, RGBColor[0.85, 0.33, 0.1]],
      {1, 11.5}, {0, -1}
    ]
  }]
];

Export["recurrent_feedback_strength_curve.png", fig1, ImageResolution -> 150];
Print["Exported: recurrent_feedback_strength_curve.png"];

(* ============================================================ *)
(* FIGURE 2: Per-Couplet Success Grid                            *)
(* X-axis: Condition (4 columns)                                 *)
(* Y-axis: Couplet ID + target word (15 rows)                    *)
(* Green = rhyme success, Red = failure                           *)
(* Highlights: couplet 1 conversion (gold border),               *)
(*             couplet 14 regression+stabilization,               *)
(*             resistant failures (4, 5, 7, 13)                  *)
(* ============================================================ *)

successColor = RGBColor[0.3, 0.75, 0.35];  (* Green *)
failureColor = RGBColor[0.9, 0.3, 0.3];    (* Red *)
convertColor = RGBColor[1.0, 0.85, 0.0];   (* Gold — for conversion highlight *)

fig2grid = Table[
  Module[{val = coupletGrid[[row, col + 2]], word = generatedWords[[row, col + 1]]},
    {If[val == 1, successColor, failureColor],
     Rectangle[{col - 1, 15 - row}, {col, 16 - row}],
     White,
     Text[
       Style[word, 10, Bold],
       {col - 0.5, 15.5 - row}
     ]}
  ],
  {row, 15}, {col, 4}
];

(* Row labels: "1. light", "2. play", ... *)
rowLabels = Table[
  Text[
    Style[
      StringJoin[ToString[coupletGrid[[row, 1]]], ". ", coupletGrid[[row, 2]]],
      11,
      If[MemberQ[{4, 5, 7, 13}, coupletGrid[[row, 1]]], Bold, Plain],
      If[MemberQ[{4, 5, 7, 13}, coupletGrid[[row, 1]]], RGBColor[0.7, 0.1, 0.1], Black]
    ],
    {-0.15, 15.5 - row}, {1, 0}
  ],
  {row, 15}
];

(* Column labels *)
colLabels = Table[
  Text[
    Style[coupletLabels[[col]], 11, Bold],
    {col - 0.5, 15.25}, {0, -1}
  ],
  {col, 4}
];

(* Conversion highlight: gold border on couplet 1, sustained s=1.0 (col 3) *)
convHighlight = {
  convertColor, AbsoluteThickness[3],
  Line[{{2, 14}, {3, 14}, {3, 15}, {2, 15}, {2, 14}}]
};

(* Regression highlight: dashed border on couplet 14, prefill s=1.0 (col 2) *)
regrHighlight = {
  RGBColor[0.6, 0.1, 0.1], AbsoluteThickness[2], Dashing[{0.02, 0.01}],
  Line[{{1, 1}, {2, 1}, {2, 2}, {1, 2}, {1, 1}}]
};

fig2 = Graphics[{
  (* Grid cells *)
  Flatten[fig2grid],
  (* Grid lines *)
  GrayLevel[0.95], AbsoluteThickness[1],
  Table[Line[{{0, y}, {4, y}}], {y, 0, 15}],
  Table[Line[{{x, 0}, {x, 15}}], {x, 0, 4}],
  (* Labels *)
  rowLabels,
  colLabels,
  (* Highlights *)
  convHighlight,
  regrHighlight
  },
  PlotLabel -> Style[
    "Per-Couplet Results: Baseline vs. Recurrent Feedback (Llama 3.2 1B)",
    14, Bold
  ],
  ImageSize -> 600,
  PlotRange -> {{-2.5, 4.2}, {-0.5, 16}},
  PlotRangePadding -> {{0, 0.2}, {0.3, 0.5}}
];

Export["recurrent_feedback_couplet_grid.png", fig2, ImageResolution -> 150];
Print["Exported: recurrent_feedback_couplet_grid.png"];
