# 绂荤嚎 `main_6class` + `Real` 娉涘寲澧炲己璁捐

## 1. 鐩爣

鏈璁＄敤浜庣畻娉曡璁¤禌閬撶殑绂荤嚎涓绘ā鍨嬫柟妗堟敹鏉燂紝鐩爣涓嶆槸缁х画浼樺寲鏃х殑 `4class` 鍩虹嚎锛岃€屾槸鍥寸粫褰撳墠宸茬粡瀹屾垚鐨?`RflyMAD v2` 鏁版嵁鍗忚锛岄噸寤轰竴濂楅潰鍚?`main_6class` 浠诲姟銆佸苟鏄惧紡鍏虫敞 `Real` 娉涘寲鑳藉姏鐨勮缁冧富绾裤€?
鏈樁娈垫牳蹇冪洰鏍囷細

- 涓讳换鍔″浐瀹氫负 `main_6class = normal / motor / propeller / sensor / wind / load`
- 涓昏瘎浠烽噸鐐瑰浐瀹氫负 `Real` 鍩熸硾鍖栵紝鑰屼笉鏄崟绾彁鍗?`SIL/HIL test`
- 鍏佽寮曞叆灏戦噺 `Real` 鏍锋湰鍙備笌璁粌/楠岃瘉锛屽苟涓?`zero-shot` 缁撴灉鍋氬姣?- 璁粌涓荤嚎浼樺厛閲囩敤 `Transformer`锛屽垱鏂扮偣鏀惧湪 `simulation-to-real` 娉涘寲绛栫暐锛岃€屼笉鏄洸鐩洿鎹?backbone

## 2. 褰撳墠浠ｇ爜涓庢暟鎹幇鐘?
褰撳墠浠撳簱宸茬粡鍏峰锛?
- `manifest_v2` 鍜?`split_v2`
- `main_6class` 鏍囩浣撶郴
- `Real` 浣滀负 `external_test` 鐨勬柊鍗忚

浣嗙绾胯缁冮摼璺粛鍋滅暀鍦ㄦ棫鏂规锛?
- [src/assets/models/train_seq_model.py](D:\UAV-Fault-Diagnosis\src\assets\\models\\train_seq_model.py) 浠嶅浐瀹氫娇鐢?`LABEL_NAMES = ['normal', 'motor', 'propeller', 'sensor']`
- [src/data/build_sequence_cache.py](D:\UAV-Fault-Diagnosis\src\data\build_sequence_cache.py) 浠嶄緷璧栨棫 `manifest/splits` 鍙ｅ緞鍜屾棫 `4class` 鏍囩鏄犲皠
- 榛樿 cache 璺緞浠嶆槸 `data/processed/seq_cache`
- 鏃?`data/processed` 鐩綍浠嶆槸鍘嗗彶浜х墿锛屼笉搴旂户缁綔涓烘柊 baseline 鐨勫敮涓€渚濇嵁

鍥犳鏈樁娈典笉鏄€滃皬淇ˉ鈥濓紝鑰屾槸瑕佸畬鎴愶細

1. `seq_cache_v2` 閲嶅缓
2. `main_6class` 璁粌鑴氭湰鍗囩骇
3. `Real few-shot + domain generalization` 瀹為獙鍗忚钀藉湴

## 3. 浠诲姟涓庤瘎浠?
### 3.1 涓讳换鍔?
- 浠诲姟鍚嶏細`main_6class`
- 鏍囩鍒楋細`label_main_6class`
- 鏍锋湰杩囨护锛歚is_in_main_6class == 1 and quality_flag != 'bad'`

### 3.2 涓昏瘎浠锋寚鏍?
涓绘寚鏍囨寜浼樺厛绾ф帓搴忥細

1. `external_test(Real) macro_f1`
2. `external_test(Real) balanced_accuracy`
3. `test(SIL/HIL) macro_f1`
4. 鍚勭被 `F1`锛岄噸鐐瑰叧娉?`sensor / wind / load`

### 3.3 鎴愬姛鏍囧噯

鏈疆涓荤嚎涓嶄互鈥滀豢鐪?test 鏇撮珮鈥濅綔涓烘垚鍔熷畾涔夛紝鑰屼互锛?
- `Real external_test` 鏄庢樉浼樹簬鏃?baseline
- `SIL/HIL test` 涓嶅嚭鐜版槑鏄鹃€€鍖?- 妯″瀷鏈哄埗鍜屽疄楠屽崗璁叿澶囧彲绛旇京鐨勫垱鏂版€?
## 4. 妯″瀷璺嚎

### 4.1 Backbone 瀵规瘮

绗竴闃舵鍏堣窇 3 涓?backbone baseline锛?
- `TextCNN`
- `CNN-LSTM`
- `Transformer`

鐩殑涓嶆槸鏈€缁堜氦浠樹笁妯″瀷锛岃€屾槸纭涓?backbone銆傛寜宸叉湁鍘嗗彶缁撴灉鍜屾枃妗ｅ垽鏂紝`Transformer` 搴斾綔涓哄悗缁寮轰富骞层€?
### 4.2 涓绘柟妗堬細`DG-Transformer`

涓绘柟妗堥噰鐢?`Transformer` 浣滀负 backbone锛屽苟鍦?`main_6class` 鍒嗙被涔嬪鍔犲叆璺ㄥ煙娉涘寲绾︽潫銆?
寤鸿缁撴瀯锛?
- Backbone锛歚Transformer`
- Head A锛歚main_6class classifier`
- Head B锛歚domain classifier`

鍩熸爣绛撅細

- `SIL`
- `HIL`
- `Real`

寤鸿鎬绘崯澶憋細

- `L_cls`: 涓诲垎绫绘崯澶?- `L_dom`: 鍩熷榻?鍩熷鎶楁崯澶?- `L = L_cls + 位 * L_dom`

鎺ㄨ崘浼樺厛瀹炵幇璺緞锛?
1. 鍏堝仛 plain `Transformer`
2. 鍐嶅仛 `Transformer + Real few-shot`
3. 鍐嶅仛 `DG-Transformer`
4. 鏈€鍚庡仛 `DG-Transformer + Real few-shot`

### 4.3 澶囬€夊寮?
濡傛灉涓绘柟妗堢ǔ瀹氬悗浠嶆湁鏃堕棿锛屽彲杩藉姞锛?
- `Hierarchical Multi-Task Transformer`
  - 鍚屾椂棰勬祴 `binary / family / main_6class`
- `TestInfo-Aware Dual-Stream`
  - 鐢?`TestInfo metadata` 鍋氶澶栧垎鏀?
浣嗚繖涓ゆ潯涓嶅簲闃诲涓绘柟妗堣惤鍦般€?
## 5. Real 鍒囧垎鍗忚

涓轰簡鍚屾椂鏀寔 `zero-shot` 鍜?`few-shot` 瀵规瘮锛宍Real` 涓嶅啀鍙繚鐣欏崟涓€ `external_test` 鍙ｅ緞銆?
寤鸿鍦ㄥ綋鍓?`Real` 鏁版嵁涓婂浐瀹氱敓鎴愪竴濂楁柊鍗忚锛?
- `Real-adapt-train`: 20%
- `Real-adapt-val`: 10%
- `Real-holdout-test`: 70%

瑕佹眰锛?
- 鎸?`label_main_6class` 鍒嗗眰鎶芥牱
- 灏介噺瑕嗙洊涓嶅悓 `flight_mode`
- 鍚屼竴 `sample_id` 涓嶅彲璺ㄩ泦鍚?- 鍥哄畾闅忔満绉嶅瓙

杩欐牱鍙互寰楀埌涓ゆ潯姝ｅ紡鍙ｅ緞锛?
- `Zero-shot`: 璁粌闃舵涓嶄娇鐢ㄤ换浣?`Real`
- `Few-shot adaptation`: 璁粌/楠岃瘉浣跨敤灏戦噺 `Real`锛屾渶缁堝湪 `Real-holdout-test` 涓婅瘎浼?
## 6. 瀹為獙鍒嗙粍

寤鸿鏈€灏戣窇浠ヤ笅 4 缁勬牳蹇冨疄楠岋細

1. `Baseline-T`
- 浠?`SIL + HIL train/val/test`
- `Real` 浠呬綔 `external_test`

2. `Real-FewShot`
- 鍦ㄨ缁?楠岃瘉涓姞鍏ュ皯閲?`Real`
- 鍦?`Real-holdout-test` 涓婅瘎浼?
3. `DG-Transformer`
- 涓嶅姞 `Real few-shot`
- 浠呴潬璺ㄥ煙绾︽潫鎻愬崌 `Real` 娉涘寲

4. `DG-Transformer + Real-FewShot`
- 鏈€缁堜富妯″瀷鍊欓€?
寤鸿琛ュ厖瀵规瘮锛?
- `CNN-LSTM` vs `TextCNN` vs `Transformer`
- `Transformer` vs `Transformer + Real-FewShot`
- `Transformer` vs `DG-Transformer`
- `DG-Transformer` vs `DG-Transformer + Real-FewShot`

## 7. cache 閲嶅缓瑕佹眰

褰撳墠 `seq_cache` 宸蹭笉閫傚悎缁х画娌跨敤锛屽繀椤婚噸寤轰负 `v2` 鏂规銆?
鏂?cache 鑷冲皯闇€瑕佹弧瓒筹細

- 杈撳叆鏉ユ簮瀵归綈 `manifest_v2 + splits_v2`
- 鏍囩鍙栬嚜 `label_main_6class`
- 鏀寔 `train / val / test / external_test`
- 杩涗竴姝ユ敮鎸?`real_adapt_train / real_adapt_val / real_holdout_test`
- 淇濈暀 `norm_stats.json`
- 鏍锋湰鍏冩暟鎹腑鑷冲皯淇濈暀锛?  - `filename`
  - `sample_id`
  - `platform`
  - `flight_mode`
  - `label_main_6class`
  - `label_id`

寤鸿鐩綍鍛藉悕锛?
- `data/cache/seq_main6_v1/`
- 鎴?`data/RflyMAD/cache/seq/main6_v1/`

涓嶈缁х画鎶婃柊 cache 鍐欏埌鏃х殑 `data/processed/seq_cache` 涓嬨€?
## 8. 璁粌浠ｇ爜鏀归€犺姹?
鐜版湁 [src/assets/models/train_seq_model.py](D:\UAV-Fault-Diagnosis\src\assets\\models\\train_seq_model.py) 闇€瑕佸崌绾э紝鑷冲皯鍖呮嫭锛?
- 鏍囩浠?`4class` 鏀逛负 `main_6class`
- 杈撳嚭鎸囨爣鏄惧紡鍖呭惈 `external_test`
- 鏀寔璇诲彇 `Real` few-shot 鍗忚
- 鏀寔 backbone 缁熶竴鍏ュ彛
- 鏀寔 domain label 杈撳叆
- 鏀寔 `plain / few-shot / DG / DG+few-shot` 妯″紡鍒囨崲

寤鸿涓嶈鍦ㄦ棫鑴氭湰涓婄户缁爢杩囧鏉′欢鍒嗘敮锛屼紭鍏堣€冭檻锛?
- 淇濈暀鏃ц剼鏈敤浜庡巻鍙插鐜?- 鏂板 `train_seq_model_v2.py` 鎴?`train_main6_dg.py`

## 9. 绛旇京鍙欎簨

璁烘枃/绛旇京涓殑鍒涙柊鐐瑰簲鑱氱劍鍦ㄢ€滆法鍩熸硾鍖栫瓥鐣モ€濓紝鑰屼笉鏄€滃爢鏇村鏉傜綉缁溾€濄€?
寤鸿琛ㄨ堪涓猴細

1. 鏋勫缓闈㈠悜 `simulation-to-real` 鐨勬棤浜烘満鏁呴殰璇婃柇璁粌妗嗘灦
2. 鍦?`main_6class` 澶氭晠闅滀换鍔′笂寮曞叆灏戦噺鐪熷疄鏍锋湰鍙備笌璺ㄥ煙澧炲己
3. 璁捐 `zero-shot -> few-shot -> domain-generalized` 鐨勭郴缁熷姣斿疄楠屽崗璁?
## 10. 瀹炴柦椤哄簭

寤鸿鎸変互涓嬮『搴忔帹杩涳細

1. 鍥哄畾 `Real` 鍒囧垎鍗忚
2. 閲嶅缓 `seq_cache_v2`
3. 璺?3 涓?backbone baseline
4. 纭涓?backbone
5. 瀹炵幇 `Real few-shot`
6. 瀹炵幇 `DG-Transformer`
7. 姹囨€?`external_test(Real)` 缁撴灉銆佹贩娣嗙煩闃靛拰鍙鍖?
## 11. 褰撳墠缁撹

褰撳墠绂荤嚎姣旇禌涓荤嚎宸茬粡鍙互鏄庣‘涓猴細

> 浣跨敤 `Transformer` 浣滀负绂荤嚎 `main_6class` 涓?backbone锛屼互 `Real` 娉涘寲涓烘牳蹇冪洰鏍囷紝鍏堥噸寤?`seq_cache_v2` 鍜岃缁冧唬鐮侊紝鍐嶉€氳繃 `few-shot Real` 涓?`domain generalization` 涓ゆ潯澧炲己璺嚎瀹屾垚鏍稿績瀵规瘮瀹為獙銆?