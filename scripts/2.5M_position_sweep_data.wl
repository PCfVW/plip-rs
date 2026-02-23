(* Position sweep data from suppress_inject_sweep on Gemma 2 2B, 2.5M CLT *)
(* Source: outputs/2.5M/suppress_inject_sweep.json *)
(* Model: google/gemma-2-2b, CLT: mntss/clt-gemma-2-2b-2.5M *)
(* Format: {{position, P(inject_word)}, ...} *)
(*                                                                            *)
(* Primary figure (closest to Anthropic's Figure 13):                        *)
(*   canL25: suppress -out, inject "can" (L25), sailor/shout prompt           *)
(*   Floor: ~2.2e-12 (flat to the mantissa), spike=0.4249, ratio=8.98e12x    *)
(*                                                                            *)
(* Secondary figure (most dramatic spike ratio):                              *)
(*   kindL25: suppress -ow, inject "kind" (L25), sailor/so prompt             *)
(*   Floor: ~5.2e-14 (flat to the mantissa), spike=0.2348, ratio=3.04e14x    *)

(* ---- Primary: suppress -out, inject "can" L25 ---- *)
(* Prompt: "A sailor sailed across the bay,\nAnd dreamed of home throughout   *)
(*          the day.\nHe raised his voice and gave a shout,\nThe truth was    *)
(*          struggling to come _"                                              *)
(* Planning site: position 34, token " " (last blank)                        *)
(* spike ratio (last_p / floor_max): 1.59e+11x                               *)
(* total ratio (last_p / baseline):  8.98e+12x                               *)

canL25Data = {
  {0,  2.2281877*^-12}, {1,  2.1628547*^-12}, {2,  2.3203860*^-12},
  {3,  2.6792060*^-12}, {4,  2.4506619*^-12}, {5,  2.5956414*^-12},
  {6,  2.4395750*^-12}, {7,  2.2326314*^-12}, {8,  2.6132868*^-12},
  {9,  2.4701218*^-12}, {10, 2.5487110*^-12}, {11, 2.3762739*^-12},
  {12, 2.5260581*^-12}, {13, 2.4015395*^-12}, {14, 2.5771427*^-12},
  {15, 2.1720483*^-12}, {16, 2.2726070*^-12}, {17, 2.1227630*^-12},
  {18, 2.5968120*^-12}, {19, 2.4695554*^-12}, {20, 2.1628532*^-12},
  {21, 2.3728164*^-12}, {22, 2.2327934*^-12}, {23, 2.2302297*^-12},
  {24, 2.1642263*^-12}, {25, 2.1115820*^-12}, {26, 2.1639238*^-12},
  {27, 2.2654313*^-12}, {28, 2.1619950*^-12}, {29, 2.3939897*^-12},
  {30, 2.2317035*^-12}, {31, 2.2299276*^-12}, {32, 2.2090743*^-12},
  {33, 2.1125198*^-12}, {34, 4.2486262*^-1}
};
canL25Tokens = {
  "<bos>", "A", " sailor", " sailed", " across", " the", " bay", ",", "\\n",
  "And", " dreamed", " of", " home", " throughout", " the", " day", ".", "\\n",
  "He", " raised", " his", " voice", " and", " gave", " a", " shout", ",", "\\n",
  "The", " truth", " was", " struggling", " to", " come", " "
};
canL25Baseline = 4.7328860*^-14;

(* ---- Secondary: suppress -ow, inject "kind" L25 ---- *)
(* Prompt: "A sailor sailed across the bay,\nAnd dreamed of home throughout   *)
(*          the day.\nThe world keeps spinning even so,\nThere is so much we  *)
(*          do not _"                                                          *)
(* Planning site: position 33, token " " (last blank)                        *)
(* spike ratio (last_p / floor_max): 3.78e+12x                               *)
(* total ratio (last_p / baseline):  3.04e+14x                               *)

kindL25Data = {
  {0,  5.4501520*^-14}, {1,  5.4083904*^-14}, {2,  5.0220650*^-14},
  {3,  5.3917486*^-14}, {4,  5.3943026*^-14}, {5,  5.5275970*^-14},
  {6,  4.7601980*^-14}, {7,  5.4602356*^-14}, {8,  5.4602983*^-14},
  {9,  5.0276078*^-14}, {10, 5.8128230*^-14}, {11, 5.3518770*^-14},
  {12, 5.4595210*^-14}, {13, 5.4154100*^-14}, {14, 5.0673120*^-14},
  {15, 5.3056250*^-14}, {16, 5.1364880*^-14}, {17, 4.6377852*^-14},
  {18, 5.0220240*^-14}, {19, 5.3947803*^-14}, {20, 5.0282730*^-14},
  {21, 5.5182353*^-14}, {22, 5.1062350*^-14}, {23, 5.3945404*^-14},
  {24, 5.7641934*^-14}, {25, 5.0065090*^-14}, {26, 6.2113200*^-14},
  {27, 5.3467772*^-14}, {28, 5.3916656*^-14}, {29, 5.3929958*^-14},
  {30, 5.3508130*^-14}, {31, 5.8747786*^-14}, {32, 4.4913292*^-14},
  {33, 2.3484705*^-1}
};
kindL25Tokens = {
  "<bos>", "A", " sailor", " sailed", " across", " the", " bay", ",", "\\n",
  "And", " dreamed", " of", " home", " throughout", " the", " day", ".", "\\n",
  "The", " world", " keeps", " spinning", " even", " so", ",", "\\n",
  "There", " is", " so", " much", " we", " do", " not", " "
};
kindL25Baseline = 7.7296796*^-16;
