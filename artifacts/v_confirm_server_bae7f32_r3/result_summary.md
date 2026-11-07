# V formal r3 results

Verdict: **NOT_CONFIRMED_KEEP_T0**; failed_criteria=['C3_r97_class_cap']

Previously used validation split, NOT an untouched held-out test. Frozen T TRADE_OFF/adopt-T0 unchanged.

## Gates
ALL_PREFLIGHT_GATES_PASS;260 passed /0 failed /0 skipped /1 existing P3 xfailed;14 unchanged goldens;9 data identities.
HEAD: bae7f3278601d0297ba307ef1c5005ebe2f41c19
Manifest LF and raw SHA: 09fdac3e1cc6e1433af770a0e9ae4281d482dcf91af6305c36f7c301bdaee858

## Per seed
|seed|V0 epoch|gate|V0 FP97|V1 FP97|ratio97|V0 FP95|V1 FP95|ratio95|V0 AP|V1 AP|wall s|
|---|---|---|---|---|---|---|---|---|---|---|---|
|42|14|PASS|785|404|0.514649682|607|255|0.420098847|0.995041753|0.997427884|995.409|
|43|78|PASS|428|413|0.964953271|282|267|0.946808511|0.997343628|0.997346439|1045.368|
|44|22|PASS|741|390|0.526315789|523|259|0.495219885|0.995406734|0.997339654|1002.823|
|45|40|N/A new seed|633|390|0.616113744|448|252|0.562500000|0.996456666|0.997580688|1053.323|
|46|16|N/A new seed|867|388|0.447520185|692|264|0.381502890|0.994557832|0.997366316|1039.680|

## C1-C4
C1 PASS: geometric ratio=0.590945708398,95% CI=[0.4580109741220865, 0.7249140317671859];5/5 ratios<1.
C2 PASS: all point recall deltas>=-1pp. C3 FAIL: GPS drop 2.109227871940pp>2pp.
|class|C2 delta pp|C2 descriptive95% CI pp|C3 drop pp|
|---|---|---|---|
|GPS|+3.935969868|[0.4650490657346418, 8.27520679074111]|+2.109227872|
|accelerometer|+3.047977422|[0.12175636255292022, 6.085860004903214]|+1.241768579|
|barometer|+1.402214022|[0.2672905209337886, 3.1834878207787565]|-0.332103321|
|gyroscope|+3.310439560|[0.46129337113541247, 5.937272057529853]|-0.425824176|
|magnetometer|+1.610352265|[0.24640295890153668, 3.2892156862745034]|-0.028756290|
|motor|+0.324067975|[0.10382299510307734, 0.5232976726935666]|-0.221314715|

C2 target/realised FP: {'42': {'threshold': 0.3362461289379905, 'target_fp': 785, 'realised_fp': 785}, '43': {'threshold': 0.475235978317319, 'target_fp': 428, 'realised_fp': 428}, '44': {'threshold': 0.34463158224305035, 'target_fp': 741, 'realised_fp': 741}, '45': {'threshold': 0.3821928712195736, 'target_fp': 633, 'realised_fp': 633}, '46': {'threshold': 0.2645844949061918, 'target_fp': 867, 'realised_fp': 867}}
C4 PASS: mean delta AP=0.001650873348,95% CI=[0.0009330626230154498, 0.0025681012717725655].
Bootstrap:2000 paired draws,278 association groups,seed20261106,0 undefined/dropped/redrawn draws.

## Descriptive
V3 FP97=377,ratio=0.949910917050,CI=[0.8755323481456102, 0.9988372637311855],5xMACs.
Natural-log FP seed SD: {'V0': {'ddof0': 0.24727293385265076, 'ddof1': 0.276459544545168}, 'V1': {'ddof0': 0.02459653083959788, 'ddof1': 0.027499757484005415}}
|seed|S best FP97|S last FP97|V0 FP97|V1 FP97|
|---|---|---|---|---|
|42|785|636|785|404|
|43|428|402|428|413|
|44|741|640|741|390|

## Integrity
Evaluation SHA256: dbfde60749af6a7ab34ea4847c45cbfb0c783f56e893b9fce35e255825832443
9135 old files and2242 sealed inputs unchanged,tracked changes0,no commit,no shutdown.

Complete deviations/failures/monitoring reconnects: execution_notes.md. No scientific execution failure/rerun;3 read-only/evidence transport resets.
