import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="🏢", layout="wide")
# The original, complex SVG animation code is preserved here as a multi-line string.
ANIMATED_SVG = """
<svg width="100%" height="100%" id="svg" viewBox="0 0 1440 690" xmlns="http://www.w3.org/2000/svg" class="transition duration-300 ease-in-out delay-150">
    <style>
        .path-0{
            animation:pathAnim-0 4s;
            animation-timing-function: linear;
            animation-iteration-count: infinite;
        }
        @keyframes pathAnim-0{
            0%{
              d: path("M 0,700 L 0,131 C 76.41148325358853,112.64593301435406 152.82296650717706,94.29186602870813 239,90 C 325.17703349282294,85.70813397129187 421.11961722488036,95.47846889952153 524,98 C 626.8803827751196,100.52153110047847 736.6985645933013,95.79425837320576 849,117 C 961.3014354066987,138.20574162679424 1076.086124401914,185.34449760765548 1175,192 C 1273.913875598086,198.65550239234452 1356.9569377990429,164.82775119617224 1440,131 L 1440,700 L 0,700 Z");
            }
            25%{
              d: path("M 0,700 L 0,131 C 121.20574162679426,97.5933014354067 242.41148325358853,64.1866028708134 329,70 C 415.58851674641147,75.8133971291866 467.5598086124402,120.84688995215312 558,122 C 648.4401913875598,123.15311004784688 777.3492822966506,80.42583732057416 871,90 C 964.6507177033494,99.57416267942584 1023.0430622009569,161.44976076555025 1112,177 C 1200.956937799043,192.55023923444975 1320.4784688995214,161.77511961722487 1440,131 L 1440,700 L 0,700 Z");
            }
            50%{
              d: path("M 0,700 L 0,131 C 115.75119617224883,165.57416267942585 231.50239234449765,200.14832535885168 322,187 C 412.49760765550235,173.85167464114832 477.7416267942583,112.98086124401914 579,103 C 680.2583732057417,93.01913875598086 817.5311004784689,133.9282296650718 902,150 C 986.4688995215311,166.0717703349282 1018.1339712918661,157.3062200956938 1099,150 C 1179.8660287081339,142.6937799043062 1309.933014354067,136.8468899521531 1440,131 L 1440,700 L 0,700 Z");
            }
            75%{
              d: path("M 0,700 L 0,131 C 115.73205741626796,137.99521531100478 231.46411483253593,144.99043062200957 327,140 C 422.5358851674641,135.00956937799043 497.87559808612434,118.03349282296651 584,98 C 670.1244019138757,77.96650717703349 767.0334928229667,54.8755980861244 856,72 C 944.9665071770333,89.1244019138756 1025.9904306220096,146.46411483253587 1122,163 C 1218.0095693779904,179.53588516746413 1329.0047846889952,155.26794258373207 1440,131 L 1440,700 L 0,700 Z");
            }
            100%{
              d: path("M 0,700 L 0,131 C 76.41148325358853,112.64593301435406 152.82296650717706,94.29186602870813 239,90 C 325.17703349282294,85.70813397129187 421.11961722488036,95.47846889952153 524,98 C 626.8803827751196,100.52153110047847 736.6985645933013,95.79425837320576 849,117 C 961.3014354066987,138.20574162679424 1076.086124401914,185.34449760765548 1175,192 C 1273.913875598086,198.65550239234452 1356.9569377990429,164.82775119617224 1440,131 L 1440,700 L 0,700 Z");
            }
        }
        defs{
            linearGradient id="gradient" x1="0%" y1="50%" x2="100%" y2="50%">
            <stop offset="5%" stop-color="#0693e3"></stop>
            <stop offset="95%" stop-color="#9900ef"></stop>
        </linearGradient>
        }
        .path-1{
            animation:pathAnim-1 4s;
            animation-timing-function: linear;
            animation-iteration-count: infinite;
        }
        @keyframes pathAnim-1{
            0%{
              d: path("M 0,700 L 0,306 C 107.55980861244021,312.01913875598086 215.11961722488041,318.0382775119618 302,307 C 388.8803827751196,295.9617224880382 455.0813397129185,267.86602870813397 563,259 C 670.9186602870815,250.13397129186606 820.5550239234451,260.4976076555024 916,257 C 1011.4449760765549,253.5023923444976 1052.6985645933014,236.14354066985646 1131,242 C 1209.3014354066986,247.85645933014354 1324.6507177033493,276.92822966507174 1440,306 L 1440,700 L 0,700 Z");
            }
            25%{
              d: path("M 0,700 L 0,306 C 67.89473684210523,304.26794258373207 135.78947368421046,302.53588516746413 244,315 C 352.21052631578954,327.46411483253587 500.73684210526324,354.1244019138756 624,360 C 747.2631578947368,365.8755980861244 845.2631578947368,350.9665071770335 917,348 C 988.7368421052632,345.0334928229665 1034.2105263157896,354.00956937799043 1117,349 C 1199.7894736842104,343.99043062200957 1319.8947368421052,324.9952153110048 1440,306 L 1440,700 L 0,700 Z");
            }
            50%{
              d: path("M 0,700 L 0,306 C 118.99521531100476,308.14354066985646 237.9904306220095,310.2870813397129 330,317 C 422.0095693779905,323.7129186602871 487.03349282296665,334.9952153110048 589,341 C 690.9665071770333,347.0047846889952 829.8755980861243,347.73205741626793 913,354 C 996.1244019138757,360.26794258373207 1023.4641148325359,372.07655502392345 1102,365 C 1180.5358851674641,357.92344497607655 1310.267942583732,331.9617224880383 1440,306 L 1440,700 L 0,700 Z");
            }
            75%{
              d: path("M 0,700 L 0,306 C 79.73205741626793,308.6794258373206 159.46411483253587,311.3588516746412 248,309 C 336.53588516746413,306.6411483253588 433.87559808612446,299.24401913875596 529,286 C 624.1244019138755,272.75598086124404 717.0334928229665,253.665071770335 823,251 C 928.9665071770335,248.334928229665 1047.9904306220096,262.09569377990425 1153,274 C 1258.0095693779904,285.90430622009575 1349.0047846889952,295.9521531100479 1440,306 L 1440,700 L 0,700 Z");
            }
            100%{
              d: path("M 0,700 L 0,306 C 107.55980861244021,312.01913875598086 215.11961722488041,318.0382775119618 302,307 C 388.8803827751196,295.9617224880382 455.0813397129185,267.86602870813397 563,259 C 670.9186602870815,250.13397129186606 820.5550239234451,260.4976076555024 916,257 C 1011.4449760765549,253.5023923444976 1052.6985645933014,236.14354066985646 1131,242 C 1209.3014354066986,247.85645933014354 1324.6507177033493,276.92822966507174 1440,306 L 1440,700 L 0,700 Z");
            }
        }
        .path-2{
            animation:pathAnim-2 4s;
            animation-timing-function: linear;
            animation-iteration-count: infinite;
        }
        @keyframes pathAnim-2{
            0%{
              d: path("M 0,700 L 0,481 C 91.79904306220095,462.2918660287081 183.5980861244019,443.5837320574163 267,434 C 350.4019138755981,424.4162679425837 425.4066985645934,423.9569377990431 533,419 C 640.5933014354066,414.0430622009569 780.7751196172248,404.58851674641147 880,431 C 979.2248803827752,457.41148325358853 1037.4928229665072,519.688995215311 1124,534 C 1210.5071770334928,548.311004784689 1325.2535885167463,514.6555023923445 1440,481 L 1440,700 L 0,700 Z");
            }
            25%{
              d: path("M 0,700 L 0,481 C 104.80382775119617,493.29665071770336 209.60765550239233,505.5933014354067 310,517 C 410.39234449760767,528.4066985645933 506.37320574162686,538.9234449760766 605,533 C 703.6267942583731,527.0765550239234 804.8995215311004,504.7129186602871 887,507 C 969.1004784688996,509.2870813397129 1032.0287081339713,536.2248803827752 1121,536 C 1209.9712918660287,535.7751196172248 1324.9856459330144,508.3875598086124 1440,481 L 1440,700 L 0,700 Z");
            }
            50%{
              d: path("M 0,700 L 0,481 C 76.2583732057416,459.1818181818182 152.5167464114832,437.3636363636364 252,430 C 351.4832535885168,422.6363636363636 474.19138755980873,429.72727272727275 577,440 C 679.8086124401913,450.27272727272725 762.7177033492823,463.72727272727275 861,469 C 959.2822966507177,474.27272727272725 1072.9377990430621,471.3636363636364 1172,472 C 1271.0622009569379,472.6363636363636 1355.531100478469,476.8181818181818 1440,481 L 1440,700 L 0,700 Z");
            }
            75%{
              d: path("M 0,700 L 0,481 C 88.66985645933013,495.7368421052631 177.33971291866027,510.4736842105263 281,492 C 384.66028708133973,473.5263157894737 503.3110047846891,421.84210526315786 601,429 C 698.6889952153109,436.15789473684214 775.4162679425835,502.1578947368421 867,511 C 958.5837320574165,519.8421052631579 1065.0239234449762,471.5263157894737 1163,457 C 1260.9760765550238,442.4736842105263 1350.4880382775118,461.7368421052631 1440,481 L 1440,700 L 0,700 Z");
            }
            100%{
              d: path("M 0,700 L 0,481 C 91.79904306220095,462.2918660287081 183.5980861244019,443.5837320574163 267,434 C 350.4019138755981,424.4162679425837 425.4066985645934,423.9569377990431 533,419 C 640.5933014354066,414.0430622009569 780.7751196172248,404.58851674641147 880,431 C 979.2248803827752,457.41148325358853 1037.4928229665072,519.688995215311 1124,534 C 1210.5071770334928,548.311004784689 1325.2535885167463,514.6555023923445 1440,481 L 1440,700 L 0,700 Z");
            }
        }
    </style>
    <defs>
        <linearGradient id="gradient" x1="0%" y1="50%" x2="100%" y2="50%">
            <stop offset="5%" stop-color="#0693e3"></stop>
            <stop offset="95%" stop-color="#9900ef"></stop>
        </linearGradient>
    </defs>
    <path d="M 0,700 L 0,131 C 76.41148325358853,112.64593301435406 152.82296650717706,94.29186602870813 239,90 C 325.17703349282294,85.70813397129187 421.11961722488036,95.47846889952153 524,98 C 626.8803827751196,100.52153110047847 736.6985645933013,95.79425837320576 849,117 C 961.3014354066987,138.20574162679424 1076.086124401914,185.34449760765548 1175,192 C 1273.913875598086,198.65550239234452 1356.9569377990429,164.82775119617224 1440,131 L 1440,700 L 0,700 Z" stroke="none" stroke-width="0" fill="url(#gradient)" fill-opacity="0.4" class="transition-all duration-300 ease-in-out delay-150 path-0"></path>
    <path d="M 0,700 L 0,306 C 107.55980861244021,312.01913875598086 215.11961722488041,318.0382775119618 302,307 C 388.8803827751196,295.9617224880382 455.0813397129185,267.86602870813397 563,259 C 670.9186602870815,250.13397129186606 820.5550239234451,260.4976076555024 916,257 C 1011.4449760765549,253.5023923444976 1052.6985645933014,236.14354066985646 1131,242 C 1209.3014354066986,247.85645933014354 1324.6507177033493,276.92822966507174 1440,306 L 1440,700 L 0,700 Z" stroke="none" stroke-width="0" fill="url(#gradient)" fill-opacity="0.53" class="transition-all duration-300 ease-in-out delay-150 path-1"></path>
    <path d="M 0,700 L 0,481 C 91.79904306220095,462.2918660287081 183.5980861244019,443.5837320574163 267,434 C 350.4019138755981,424.4162679425837 425.4066985645934,423.9569377990431 533,419 C 640.5933014354066,414.0430622009569 780.7751196172248,404.58851674641147 880,431 C 979.2248803827752,457.41148325358853 1037.4928229665072,519.688995215311 1124,534 C 1210.5071770334928,548.311004784689 1325.2535885167463,514.6555023923445 1440,481 L 1440,700 L 0,700 Z" stroke="none" stroke-width="0" fill="url(#gradient)" fill-opacity="1" class="transition-all duration-300 ease-in-out delay-150 path-2"></path>
</svg>
"""

# 1. Embed the SVG animation directly using st.markdown
# The wrapper div ensures the animation covers the entire viewport and stays behind the content.
st.markdown(
    f"""
    <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100vh; z-index: -1; overflow: hidden;">
        <svg style="width: 100%; height: 100%; object-fit: cover; min-height: 100vh;" viewBox="0 0 1440 700" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
            <style>
                .path-0 {{
                    animation: pathAnim-0 4s;
                    animation-timing-function: linear;
                    animation-iteration-count: infinite;
                }}
                .path-1 {{
                    animation: pathAnim-1 4s;
                    animation-timing-function: linear;
                    animation-iteration-count: infinite;
                }}
                .path-2 {{
                    animation: pathAnim-2 4s;
                    animation-timing-function: linear;
                    animation-iteration-count: infinite;
                }}
                @keyframes pathAnim-0 {{
                    0%{{
                        d: path("M 0,700 L 0,131 C 76.41148325358853,112.64593301435406 152.82296650717706,94.29186602870813 239,90 C 325.17703349282294,85.70813397129187 421.11961722488036,95.47846889952153 524,98 C 626.8803827751196,100.52153110047847 736.6985645933013,95.79425837320576 849,117 C 961.3014354066987,138.20574162679424 1076.086124401914,185.34449760765548 1175,192 C 1273.913875598086,198.65550239234452 1356.9569377990429,164.82775119617224 1440,131 L 1440,700 L 0,700 Z");
                    }}
                    25%{{
                        d: path("M 0,700 L 0,131 C 95.55023923444976,149.8564593301436 191.10047846889952,168.71291866028708 279,162 C 366.89952153110048,155.28708133971292 447.1483253588517,123.00478468899522 549,118 C 650.8516746411483,112.99521531100478 774.3062200956938,135.26794258373206 876,145 C 977.6937799043062,154.73205741626794 1057.6267942583732,151.92344497607656 1155,155 C 1252.3732057416268,158.07655502392344 1367.1866028708134,167.03827751196172 1440,131 L 1440,700 L 0,700 Z");
                    }}
                    50%{{
                        d: path("M 0,700 L 0,131 C 88.66985645933013,149.8564593301436 177.33971291866027,168.71291866028708 281,162 C 384.66028708133973,155.28708133971292 503.3110047846891,123.00478468899522 601,118 C 698.6889952153109,112.99521531100478 775.4162679425835,135.26794258373206 867,145 C 958.5837320574165,154.73205741626794 1065.0239234449762,151.92344497607656 1163,155 C 1260.9760765550238,158.07655502392344 1350.4880382775118,167.03827751196172 1440,131 L 1440,700 L 0,700 Z");
                    }}
                    75%{{
                        d: path("M 0,700 L 0,131 C 91.79904306220095,112.64593301435406 183.5980861244019,94.29186602870813 267,90 C 350.4019138755981,85.70813397129187 425.4066985645934,95.47846889952153 533,98 C 640.5933014354066,100.52153110047847 780.7751196172248,95.79425837320576 880,117 C 979.2248803827752,138.20574162679424 1037.4928229665072,185.34449760765548 1124,192 C 1210.5071770334928,198.65550239234452 1325.2535885167463,164.82775119617224 1440,131 L 1440,700 L 0,700 Z");
                    }}
                    100%{{
                        d: path("M 0,700 L 0,131 C 76.41148325358853,112.64593301435406 152.82296650717706,94.29186602870813 239,90 C 325.17703349282294,85.70813397129187 421.11961722488036,95.47846889952153 524,98 C 626.8803827751196,100.52153110047847 736.6985645933013,95.79425837320576 849,117 C 961.3014354066987,138.20574162679424 1076.086124401914,185.34449760765548 1175,192 C 1273.913875598086,198.65550239234452 1356.9569377990429,164.82775119617224 1440,131 L 1440,700 L 0,700 Z");
                    }}
                }}
                @keyframes pathAnim-1 {{
                    0%{{
                        d: path("M 0,700 L 0,306 C 107.55980861244021,312.01913875598086 215.11961722488041,318.0382775119618 302,307 C 388.8803827751196,295.9617224880382 455.0813397129185,267.86602870813397 563,259 C 670.9186602870815,250.13397129186606 820.5550239234451,260.4976076555024 916,257 C 1011.4449760765549,253.5023923444976 1052.6985645933014,236.14354066985646 1131,242 C 1209.3014354066986,247.85645933014354 1324.6507177033493,276.92822966507174 1440,306 L 1440,700 L 0,700 Z");
                    }}
                    25%{{
                        d: path("M 0,700 L 0,306 C 95.55023923444976,324.8564593301436 191.10047846889952,343.71291866028708 279,337 C 366.89952153110048,330.28708133971292 447.1483253588517,298.00478468899522 549,293 C 650.8516746411483,287.99521531100478 774.3062200956938,310.26794258373206 876,320 C 977.6937799043062,329.73205741626794 1057.6267942583732,326.92344497607656 1155,330 C 1252.3732057416268,333.07655502392344 1367.1866028708134,342.03827751196172 1440,306 L 1440,700 L 0,700 Z");
                    }}
                    50%{{
                        d: path("M 0,700 L 0,306 C 88.66985645933013,324.8564593301436 177.33971291866027,343.71291866028708 281,337 C 384.66028708133973,330.28708133971292 503.3110047846891,298.00478468899522 601,293 C 698.6889952153109,287.99521531100478 775.4162679425835,310.26794258373206 867,320 C 958.5837320574165,329.73205741626794 1065.0239234449762,326.92344497607656 1163,330 C 1260.9760765550238,333.07655502392344 1350.4880382775118,342.03827751196172 1440,306 L 1440,700 L 0,700 Z");
                    }}
                    75%{{
                        d: path("M 0,700 L 0,306 C 91.79904306220095,287.64593301435406 183.5980861244019,269.29186602870813 267,265 C 350.4019138755981,260.70813397129187 425.4066985645934,270.47846889952153 533,273 C 640.5933014354066,275.52153110047847 780.7751196172248,270.79425837320576 880,292 C 979.2248803827752,313.20574162679424 1037.4928229665072,360.34449760765548 1124,367 C 1210.5071770334928,373.65550239234452 1325.2535885167463,339.82775119617224 1440,306 L 1440,700 L 0,700 Z");
                    }}
                    100%{{
                        d: path("M 0,700 L 0,306 C 107.55980861244021,312.01913875598086 215.11961722488041,318.0382775119618 302,307 C 388.8803827751196,295.9617224880382 455.0813397129185,267.86602870813397 563,259 C 670.9186602870815,250.13397129186606 820.5550239234451,260.4976076555024 916,257 C 1011.4449760765549,253.5023923444976 1052.6985645933014,236.14354066985646 1131,242 C 1209.3014354066986,247.85645933014354 1324.6507177033493,276.92822966507174 1440,306 L 1440,700 L 0,700 Z");
                    }}
                }}
                @keyframes pathAnim-2 {{
                    0%{{
                        d: path("M 0,700 L 0,481 C 91.79904306220095,462.2918660287081 183.5980861244019,443.5837320574163 267,434 C 350.4019138755981,424.4162679425837 425.4066985645934,423.9569377990431 533,419 C 640.5933014354066,414.0430622009569 780.7751196172248,404.58851674641147 880,431 C 979.2248803827752,457.41148325358853 1037.4928229665072,519.688995215311 1124,534 C 1210.5071770334928,548.311004784689 1325.2535885167463,514.6555023923445 1440,481 L 1440,700 L 0,700 Z");
                    }}
                    25%{{
                        d: path("M 0,700 L 0,481 C 95.55023923444976,499.8564593301436 191.10047846889952,518.7129186602871 279,512 C 366.89952153110048,505.28708133971292 447.1483253588517,473.00478468899522 549,468 C 650.8516746411483,462.99521531100478 774.3062200956938,485.26794258373206 876,495 C 977.6937799043062,504.73205741626794 1057.6267942583732,501.92344497607656 1155,505 C 1252.3732057416268,508.07655502392344 1367.1866028708134,517.0382775119617 1440,481 L 1440,700 L 0,700 Z");
                    }}
                    50%{{
                        d: path("M 0,700 L 0,481 C 88.66985645933013,499.8564593301436 177.33971291866027,518.7129186602871 281,512 C 384.66028708133973,505.28708133971292 503.3110047846891,473.00478468899522 601,468 C 698.6889952153109,462.99521531100478 775.4162679425835,485.26794258373206 867,495 C 958.5837320574165,504.73205741626794 1065.0239234449762,501.92344497607656 1163,505 C 1260.9760765550238,508.07655502392344 1350.4880382775118,517.0382775119617 1440,481 L 1440,700 L 0,700 Z");
                    }}
                    75%{{
                        d: path("M 0,700 L 0,481 C 91.79904306220095,462.64593301435406 183.5980861244019,444.29186602870813 267,440 C 350.4019138755981,435.70813397129187 425.4066985645934,445.47846889952153 533,448 C 640.5933014354066,450.52153110047847 780.7751196172248,445.79425837320576 880,467 C 979.2248803827752,488.20574162679424 1037.4928229665072,535.3444976076555 1124,542 C 1210.5071770334928,548.6555023923445 1325.2535885167463,514.8277511961722 1440,481 L 1440,700 L 0,700 Z");
                    }}
                    100%{{
                        d: path("M 0,700 L 0,481 C 91.79904306220095,462.2918660287081 183.5980861244019,443.5837320574163 267,434 C 350.4019138755981,424.4162679425837 425.4066985645934,423.9569377990431 533,419 C 640.5933014354066,414.0430622009569 780.7751196172248,404.58851674641147 880,431 C 979.2248803827752,457.41148325358853 1037.4928229665072,519.688995215311 1124,534 C 1210.5071770334928,548.311004784689 1325.2535885167463,514.6555023923445 1440,481 L 1440,700 L 0,700 Z");
                    }}
                }}
            </style>
            <defs>
                <linearGradient id="gradient" x1="0%" y1="50%" x2="100%" y2="50%">
                    <stop offset="5%" stop-color="#0693e3"></stop>
                    <stop offset="95%" stop-color="#9900ef"></stop>
                </linearGradient>
            </defs>
            <path d="M 0,700 L 0,131 C 76.41148325358853,112.64593301435406 152.82296650717706,94.29186602870813 239,90 C 325.17703349282294,85.70813397129187 421.11961722488036,95.47846889952153 524,98 C 626.8803827751196,100.52153110047847 736.6985645933013,95.79425837320576 849,117 C 961.3014354066987,138.20574162679424 1076.086124401914,185.34449760765548 1175,192 C 1273.913875598086,198.65550239234452 1356.9569377990429,164.82775119617224 1440,131 L 1440,700 L 0,700 Z" stroke="none" stroke-width="0" fill="url(#gradient)" fill-opacity="0.4" class="transition-all duration-300 ease-in-out delay-150 path-0"></path>
            <path d="M 0,700 L 0,306 C 107.55980861244021,312.01913875598086 215.11961722488041,318.0382775119618 302,307 C 388.8803827751196,295.9617224880382 455.0813397129185,267.86602870813397 563,259 C 670.9186602870815,250.13397129186606 820.5550239234451,260.4976076555024 916,257 C 1011.4449760765549,253.5023923444976 1052.6985645933014,236.14354066985646 1131,242 C 1209.3014354066986,247.85645933014354 1324.6507177033493,276.92822966507174 1440,306 L 1440,700 L 0,700 Z" stroke="none" stroke-width="0" fill="url(#gradient)" fill-opacity="0.53" class="transition-all duration-300 ease-in-out delay-150 path-1"></path>
            <path d="M 0,700 L 0,481 C 91.79904306220095,462.2918660287081 183.5980861244019,443.5837320574163 267,434 C 350.4019138755981,424.4162679425837 425.4066985645934,423.9569377990431 533,419 C 640.5933014354066,414.0430622009569 780.7751196172248,404.58851674641147 880,431 C 979.2248803827752,457.41148325358853 1037.4928229665072,519.688995215311 1124,534 C 1210.5071770334928,548.311004784689 1325.2535885167463,514.6555023923445 1440,481 L 1440,700 L 0,700 Z" stroke="none" stroke-width="0" fill="url(#gradient)" fill-opacity="1" class="transition-all duration-300 ease-in-out delay-150 path-2"></path>
        </svg>
    </div>
    """,
    unsafe_allow_html=True
)

# 2. Add custom CSS to style the Streamlit containers for readability
st.markdown("""
<style>
/* Mobile-first responsive design */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem !important;
        margin-top: 1rem !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
        max-width: 100% !important;
    }
    
    /* Ensure SVG covers full mobile screen height */
    div[style*="position: fixed"] svg {
        min-height: 100vh !important;
        min-height: 100dvh !important; /* Dynamic viewport height for mobile */
    }
}

/* Ensure the main app content area has a solid/translucent background 
   so the text doesn't clash with the animated SVG behind it. */
.main .block-container {
    background: rgba(0, 0, 0, 0.6); /* Semi-transparent background for content */
    border-radius: 10px;
    padding: 6rem;
    margin-top: 4rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

/* Make the sidebar transparent */
.stSidebar > div:first-child {
    background: rgba(0, 0, 0, 0.6) !important;
}

.css-1d391kg {
    background: rgba(0, 0, 0, 0.6) !important;
}

.css-1lcbmhc {
    background: rgba(0, 0, 0, 0.6) !important;
}

.css-17eq0hr {
    background: rgba(0, 0, 0, 0.6) !important;
}

.stSidebar {
    background: rgba(0, 0, 0, 0.6) !important;
}

section[data-testid="stSidebar"] {
    background: rgba(0, 0, 0, 0.9) !important;
}

section[data-testid="stSidebar"] > div {
    background: rgba(240, 242, 246, 0.1) !important;
}

/* Ensure the app itself has no background color (transparent) so the SVG shows */
.stApp {
    background-color: transparent;
    min-height: 100vh;
    min-height: 100dvh; /* Dynamic viewport height for mobile */
}

/* Set the title color to match the theme */
h1 {
    text-align: center;
    color: #ffffff;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
}


/* Ensure proper viewport handling */
html, body {
    height: 100%;
    min-height: 100vh;
    min-height: 100dvh;
}

/* Keep header visible but transparent */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: auto !important;
    min-height: 2.875rem !important;
}

/* Ensure sidebar toggle button is always visible */
button[data-testid="collapsedControl"],
button[kind="header"],
.css-1rs6os button,
.css-vk3wp9 button {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: rgba(255, 255, 255, 0.15) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 6px !important;
    color: white !important;
    padding: 0.25rem 0.5rem !important;
    margin: 0.5rem !important;
    z-index: 999999 !important;
    position: relative !important;
}

button[data-testid="collapsedControl"]:hover,
button[kind="header"]:hover,
.css-1rs6os button:hover,
.css-vk3wp9 button:hover {
    background: rgba(255, 255, 255, 0.25) !important;
    transform: scale(1.05) !important;
    transition: all 0.2s ease !important;
}

/* Force show hamburger icon */
.css-1rs6os,
.css-vk3wp9 {
    display: block !important;
    visibility: visible !important;
}

.stDeployButton {
    display: none;
}

footer {
    display: none;
}

.stDecoration {
    display: none;
}

#MainMenu {
    visibility: hidden;
}

footer:after {
    content: '';
    display: none;
}

.viewerBadge_container__1QSob {
    display: none;
}

.viewerBadge_link__1S137 {
    display: none;
}

/* Hide GitHub profile link in header */
a[href*="github.com"] {
    display: none !important;
}

/* Hide all header links except sidebar toggle */
header[data-testid="stHeader"] a {
    display: none !important;
}

/* Hide toolbar items */
.css-14xtw13 {
    display: none !important;
}

.css-1544g2n {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_train_model():
    """Load data and train model"""
    # Load and preprocess data (same as final_model.py)
    df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    df = df.drop_duplicates()
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    df = df.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours'], axis=1, errors='ignore')
    
    # Handle categorical variables
    categorical_cols = ['Department', 'Gender', 'OverTime', 'BusinessTravel', 'EducationField', 'JobRole', 'MaritalStatus']
    for col in categorical_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
    
    # Handle remaining object columns
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in object_cols:
        df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
    
    # Prepare features
    X = df.drop('Attrition', axis=1).astype(float)
    y = df['Attrition'].astype(int)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns.tolist(), df

def main():
    st.title("Employee Attrition Prediction System")
   # st.markdown("**Bonus Task: Interactive Employee Attrition Predictor**")
    
    # Load model and data
    model, scaler, feature_names, df = load_and_train_model()
    
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Dataset Overview", "Model Insights"])
    
    if page == "Prediction":
        st.header("Predict Employee Attrition")
        st.markdown("Enter employee details to predict attrition risk:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Information")
            age = st.slider("Age", 18, 65, 30)
            monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5000)
            distance_from_home = st.slider("Distance from Home (miles)", 1, 30, 10)
            total_working_years = st.slider("Total Working Years", 0, 40, 10)
            years_at_company = st.slider("Years at Company", 0, 40, 5)
        
        with col2:
            st.subheader("Job & Satisfaction")
            job_satisfaction = st.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4], index=2)
            environment_satisfaction = st.selectbox("Environment Satisfaction (1-4)", [1, 2, 3, 4], index=2)
            work_life_balance = st.selectbox("Work Life Balance (1-4)", [1, 2, 3, 4], index=2)
            overtime = st.selectbox("Overtime", ["No", "Yes"])
            job_level = st.slider("Job Level", 1, 5, 2)
        
        if st.button("Predict Attrition Risk", type="primary"):
            # Create input with average values for missing features
            input_data = np.zeros(len(feature_names))
            
            # Map known inputs to feature positions
            feature_mapping = {
                'Age': age,
                'MonthlyIncome': monthly_income,
                'DistanceFromHome': distance_from_home,
                'TotalWorkingYears': total_working_years,
                'YearsAtCompany': years_at_company,
                'JobSatisfaction': job_satisfaction,
                'EnvironmentSatisfaction': environment_satisfaction,
                'WorkLifeBalance': work_life_balance,
                'JobLevel': job_level,
                'DailyRate': 800,  # Default
                'HourlyRate': 65,  # Default
                'MonthlyRate': 14000,  # Default
                'Education': 3,  # Default
                'JobInvolvement': 3,  # Default
                'NumCompaniesWorked': 2,  # Default
                'PercentSalaryHike': 15,  # Default
                'PerformanceRating': 3,  # Default
                'RelationshipSatisfaction': 3,  # Default
                'StockOptionLevel': 1,  # Default
                'TrainingTimesLastYear': 3,  # Default
                'YearsInCurrentRole': min(years_at_company, 4),
                'YearsSinceLastPromotion': max(0, years_at_company - 2),
                'YearsWithCurrManager': min(years_at_company, 3)
            }
            
            # Fill in the input array
            for i, feature in enumerate(feature_names):
                if feature in feature_mapping:
                    input_data[i] = feature_mapping[feature]
                elif feature == f'OverTime_Yes':
                    input_data[i] = 1 if overtime == 'Yes' else 0
                else:
                    # Set default values for encoded categorical features
                    input_data[i] = 0
            
            # Scale the input
            input_scaled = scaler.transform([input_data])
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error(f"**HIGH RISK** of Attrition")
                    st.error(f"Probability: {probability[1]:.1%}")
                else:
                    st.success(f"**LOW RISK** of Attrition")
                    st.success(f"Probability: {probability[0]:.1%}")
            
            with col2:
                st.subheader("Risk Factors")
                risk_factors = []
                if monthly_income < 5000:
                    risk_factors.append("💰 Low monthly income")
                if age < 30:
                    risk_factors.append("👶 Young age")
                if overtime == "Yes":
                    risk_factors.append("⏰ Overtime work")
                if distance_from_home > 15:
                    risk_factors.append("🚗 Long commute")
                if job_satisfaction <= 2:
                    risk_factors.append("😞 Low job satisfaction")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.warning(factor)
                else:
                    st.info("✨ No major risk factors identified")
    
    elif page == "Dataset Overview":
        st.header("Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Employees", len(df))
        with col2:
            attrition_rate = df['Attrition'].mean() * 100
            st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
        with col3:
            st.metric("Features", len(feature_names))
        
        # Show attrition distribution
        st.subheader("Attrition Distribution")
        attrition_counts = df['Attrition'].value_counts()
        st.bar_chart(attrition_counts)
        
        # Show sample data
        st.subheader("Sample Data")
        st.dataframe(df.head())
    
    elif page == "Model Insights":
        st.header("Model Insights")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        st.subheader("Top 15 Most Important Features")
        top_features = feature_importance.head(15)
        st.bar_chart(top_features.set_index('feature')['importance'])
        
        st.subheader("Feature Importance Table")
        st.dataframe(feature_importance.head(20))
        
        # Model performance info
        st.subheader("Model Information")
        st.info(f"""
        **Model Type**: Random Forest with Balanced Class Weights
        **Number of Trees**: 100
        **Features Used**: {len(feature_names)}
        **Training Strategy**: Handles class imbalance with balanced weights
        """)

if __name__ == "__main__":
    main()