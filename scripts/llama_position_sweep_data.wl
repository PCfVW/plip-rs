(* Position sweep data from suppress_inject_sweep on Llama 3.2 1B *)
(* Source: outputs/suppress_inject_sweep_llama_v2.json *)
(* Prompts: corpus/llama_prompts.json *)
(* Format: {{position, P(inject_word)}, ...} *)

(* -ee prompt -> inject "that" (L14:13043), baseline=5.81e-6, max=0.777 at pos 30, ratio=133879x *)
eeThat = {
  {0, 5.118114*^-6}, {1, 4.224475*^-6}, {2, 5.453728*^-6}, {3, 7.945997*^-6},
  {4, 6.6262387*^-6}, {5, 6.54433*^-6}, {6, 6.0848743*^-6}, {7, 6.152764*^-6},
  {8, 6.57928*^-6}, {9, 7.097447*^-6}, {10, 1.8752731*^-5}, {11, 9.885309*^-6},
  {12, 6.1060323*^-6}, {13, 6.4470446*^-6}, {14, 6.583036*^-6}, {15, 6.845204*^-6},
  {16, 6.6092334*^-6}, {17, 6.41665*^-6}, {18, 5.7472485*^-6}, {19, 6.0991447*^-6},
  {20, 6.518313*^-6}, {21, 6.9799225*^-6}, {22, 6.194547*^-6}, {23, 6.612303*^-6},
  {24, 6.184173*^-6}, {25, 5.8015116*^-6}, {26, 6.0684956*^-6}, {27, 6.157925*^-6},
  {28, 7.3344195*^-6}, {29, 9.694057*^-5}, {30, 0.77701163}
};
eeThatTokens = {"<|begin_of_text|>", "The", " birds", " were", " singing", " in",
  " the", " tree", ",\\n", "And", " everything", " was", " wild", " and", " free",
  ".\\n", "The", " river", " ran", " down", " to", " the", " sea", ",\\n",
  "There", " is", " so", " much", " we", " cannot", " "};
eeThatBaseline = 5.805407*^-6;

(* -oo prompt -> inject "that" (L14:13043), baseline=1.96e-7, max=0.452 at pos 30, ratio=2304009x *)
ooThat = {
  {0, 1.8493736*^-7}, {1, 1.09805605*^-7}, {2, 1.7421118*^-7}, {3, 1.9711645*^-7},
  {4, 1.9147409*^-7}, {5, 1.7734298*^-7}, {6, 2.008822*^-7}, {7, 1.338996*^-7},
  {8, 1.8255484*^-7}, {9, 2.1721945*^-7}, {10, 2.3079863*^-7}, {11, 2.819098*^-7},
  {12, 1.8988779*^-7}, {13, 2.71439*^-7}, {14, 1.8094585*^-7}, {15, 1.9414794*^-7},
  {16, 2.1931967*^-7}, {17, 1.9228068*^-7}, {18, 2.1877177*^-7}, {19, 1.9837307*^-7},
  {20, 2.0236459*^-7}, {21, 1.7753865*^-7}, {22, 2.0282208*^-7}, {23, 2.2067536*^-7},
  {24, 1.6353827*^-7}, {25, 1.7900751*^-7}, {26, 1.810004*^-7}, {27, 1.9237889*^-7},
  {28, 2.5727238*^-7}, {29, 1.4074762*^-6}, {30, 0.45185935}
};
ooThatTokens = {"<|begin_of_text|>", "The", " morning", " sky", " was", " painted",
  " blue", ",\\n", "The", " garden", " spark", "led", " bright", " with", " dew",
  ".\\n", "The", " world", " had", " started", " fresh", " and", " new", ",\\n",
  "And", " there", " was", " nothing", " left", " to", " "};
ooThatBaseline = 1.960986*^-7;

(* -ore prompt -> inject "that" (L14:13043), baseline=3.01e-6, max=0.320 at pos 31, ratio=106081x *)
oreThat = {
  {0, 3.0608219*^-6}, {1, 3.0398144*^-6}, {2, 1.6816641*^-6}, {3, 3.79264*^-6},
  {4, 4.040128*^-6}, {5, 6.7705*^-6}, {6, 3.6652061*^-6}, {7, 3.399783*^-6},
  {8, 5.818201*^-6}, {9, 3.1004636*^-6}, {10, 3.3356193*^-6}, {11, 4.964077*^-6},
  {12, 8.815289*^-6}, {13, 3.6384151*^-6}, {14, 4.1456283*^-6}, {15, 4.0214386*^-6},
  {16, 3.0242218*^-6}, {17, 5.141961*^-6}, {18, 3.5049127*^-6}, {19, 3.5537148*^-6},
  {20, 3.459495*^-6}, {21, 3.480444*^-6}, {22, 4.5318106*^-6}, {23, 2.8783236*^-6},
  {24, 3.720608*^-6}, {25, 3.7550813*^-6}, {26, 5.707843*^-6}, {27, 3.019833*^-6},
  {28, 3.7133261*^-6}, {29, 4.1922335*^-6}, {30, 1.5417338*^-5}, {31, 0.3196224}
};
oreThatTokens = {"<|begin_of_text|>", "The", " waves", " came", " crashing", " on",
  " the", " shore", ",\\n", "The", " wind", " was", " how", "ling", " more", " and",
  " more", ".\\n", "She", " asked", " what", " all", " the", " fuss", " was", " for",
  ",\\n", "And", " opened", " up", " the", " "};
oreThatBaseline = 3.0138401*^-6;

(* -at prompt -> inject "for" (L1:5297), baseline=3.64e-6, max=0.0089 at pos 32, ratio=2447x *)
atFor = {
  {0, 3.720197*^-6}, {1, 3.9518686*^-6}, {2, 3.4790205*^-6}, {3, 3.4972481*^-6},
  {4, 3.2894097*^-6}, {5, 3.852659*^-6}, {6, 3.8036926*^-6}, {7, 4.541176*^-6},
  {8, 3.7726668*^-6}, {9, 8.449317*^-6}, {10, 3.4553882*^-6}, {11, 2.9544642*^-6},
  {12, 3.6062122*^-6}, {13, 5.0718822*^-6}, {14, 4.2258253*^-6}, {15, 3.6793112*^-6},
  {16, 8.051285*^-6}, {17, 4.3261894*^-6}, {18, 3.2209866*^-6}, {19, 3.6511171*^-6},
  {20, 4.0249497*^-6}, {21, 3.5495314*^-6}, {22, 3.3791994*^-6}, {23, 3.640789*^-6},
  {24, 3.2798184*^-6}, {25, 3.9601946*^-6}, {26, 2.486812*^-6}, {27, 3.573887*^-6},
  {28, 3.158533*^-6}, {29, 4.8663933*^-6}, {30, 7.3150163*^-6}, {31, 3.631117*^-6},
  {32, 8.8989595*^-3}
};
atForTokens = {"<|begin_of_text|>", "The", " old", " man", " wore", " a", " t",
  "attered", " hat", ",\\n", "Upon", " the", " porch", " he", " always", " sat",
  ".\\n", "He", " told", " the", " tale", " of", " this", " and", " that", ",\\n",
  "And", " in", " the", " corner", " slept", " the", " "};
atForBaseline = 3.637762*^-6;

(* -at prompt -> inject "will" (L13:26263), baseline=1.95e-6, max=0.0076 at pos 32, ratio=3907x *)
atWill = {
  {0, 1.8306692*^-6}, {1, 2.0144114*^-6}, {2, 2.0374366*^-6}, {3, 2.0073246*^-6},
  {4, 1.8836079*^-6}, {5, 1.9336026*^-6}, {6, 1.8822202*^-6}, {7, 1.8521471*^-6},
  {8, 1.9713177*^-6}, {9, 1.8282275*^-6}, {10, 1.8259826*^-6}, {11, 2.0168193*^-6},
  {12, 1.8098542*^-6}, {13, 1.9866661*^-6}, {14, 1.8240095*^-6}, {15, 1.956603*^-6},
  {16, 1.9927168*^-6}, {17, 1.960593*^-6}, {18, 1.9752654*^-6}, {19, 1.9737251*^-6},
  {20, 1.8782869*^-6}, {21, 1.9654722*^-6}, {22, 1.8177224*^-6}, {23, 1.8719983*^-6},
  {24, 1.9786830*^-6}, {25, 2.0614686*^-6}, {26, 2.0690002*^-6}, {27, 2.0175585*^-6},
  {28, 1.9471036*^-6}, {29, 2.1735837*^-6}, {30, 2.321618*^-6}, {31, 1.5514622*^-6},
  {32, 7.606709*^-3}
};
atWillTokens = atForTokens;
atWillBaseline = 1.947154*^-6;

(* -at prompt -> inject "are" (L6:14873), baseline=1.61e-6, max=0.0073 at pos 32, ratio=4505x *)
atAre = {
  {0, 1.5984468*^-6}, {1, 1.6140804*^-6}, {2, 1.6445742*^-6}, {3, 1.7897235*^-6},
  {4, 1.5908223*^-6}, {5, 1.9432327*^-6}, {6, 1.7599695*^-6}, {7, 1.8753443*^-6},
  {8, 1.9390673*^-6}, {9, 1.6327639*^-6}, {10, 1.8140258*^-6}, {11, 1.7062041*^-6},
  {12, 1.6350988*^-6}, {13, 1.6021472*^-6}, {14, 1.7787839*^-6}, {15, 1.7050806*^-6},
  {16, 1.7380098*^-6}, {17, 1.5886542*^-6}, {18, 1.6615586*^-6}, {19, 1.6080162*^-6},
  {20, 1.5987782*^-6}, {21, 1.7263532*^-6}, {22, 1.914693*^-6}, {23, 1.6343453*^-6},
  {24, 1.6372786*^-6}, {25, 1.7288388*^-6}, {26, 1.4426093*^-6}, {27, 1.6749417*^-6},
  {28, 1.6252695*^-6}, {29, 1.6691553*^-6}, {30, 1.7194071*^-6}, {31, 9.859164*^-7},
  {32, 7.271887*^-3}
};
atAreTokens = atForTokens;
atAreBaseline = 1.614247*^-6;

(* ============================================================ *)
(* PLOTTING CODE — Llama 3.2 1B Position Sweep Figures          *)
(* Generates PNGs matching the melometis figure style           *)
(* ============================================================ *)

(* Helper: position sweep plot with log-scale y-axis *)
positionSweepPlot[data_, tokens_, baseline_, title_, filename_] :=
  Module[{n = Length[data], logData, maxP, maxPos, ratio, plot},
    logData = {#[[1]], Log10[#[[2]]]} & /@ data;
    maxP = Max[data[[All, 2]]];
    maxPos = data[[Position[data[[All, 2]], maxP][[1, 1]], 1]];
    ratio = Round[maxP / baseline];
    plot = Show[
      ListLinePlot[logData,
        PlotStyle -> {Thick, RGBColor[0.2, 0.4, 0.8]},
        PlotMarkers -> {"\[FilledCircle]", 6},
        Frame -> True,
        FrameLabel -> {
          Style["Token position", 14],
          Style["log\[ThinSpace]P(inject word)", 14, Italic]
        },
        PlotLabel -> Style[title, 16, Bold],
        PlotRange -> {Automatic, {Floor[Min[logData[[All, 2]]]] - 0.5, 0.2}},
        GridLines -> {None, Automatic},
        GridLinesStyle -> Directive[GrayLevel[0.85]],
        ImageSize -> 700,
        AspectRatio -> 0.45,
        PlotRangePadding -> {{Scaled[0.02], Scaled[0.02]}, {0.3, 0.3}}
      ],
      (* Annotate the spike *)
      Graphics[{
        RGBColor[0.9, 0.2, 0.2],
        PointSize[0.015], Point[{maxPos, Log10[maxP]}],
        Text[
          Style[
            StringJoin["P = ", ToString[NumberForm[maxP, {3, 3}]],
              "\n(", ToString[ratio], "\[Times] baseline)"],
            12, Bold, RGBColor[0.9, 0.2, 0.2]
          ],
          {maxPos - 2, Log10[maxP] + 0.3}, {1, -1}
        ]
      }],
      (* Baseline annotation *)
      Graphics[{
        Dashed, GrayLevel[0.5],
        Line[{{0, Log10[baseline]}, {n, Log10[baseline]}}],
        Text[
          Style["baseline", 10, Italic, GrayLevel[0.4]],
          {2, Log10[baseline] + 0.2}, {-1, -1}
        ]
      }]
    ];
    Export[filename, plot, ImageResolution -> 150];
    plot
  ];

(* --- Generate the three key figures --- *)

(* Figure 1: -ee -> "that" (strongest: P = 0.777, 133,879x) *)
positionSweepPlot[eeThat, eeThatTokens, eeThatBaseline,
  "Llama 3.2 1B: suppress -ee, inject \"that\" (L14:13043)",
  "llama_sweep_ee_that.png"
];

(* Figure 2: -oo -> "that" (highest ratio: 2,304,009x) *)
positionSweepPlot[ooThat, ooThatTokens, ooThatBaseline,
  "Llama 3.2 1B: suppress -oo, inject \"that\" (L14:13043)",
  "llama_sweep_oo_that.png"
];

(* Figure 3: -ore -> "that" (106,081x) *)
positionSweepPlot[oreThat, oreThatTokens, oreThatBaseline,
  "Llama 3.2 1B: suppress -ore, inject \"that\" (L14:13043)",
  "llama_sweep_ore_that.png"
];

Print["Exported: llama_sweep_ee_that.png"];
Print["Exported: llama_sweep_oo_that.png"];
Print["Exported: llama_sweep_ore_that.png"];
