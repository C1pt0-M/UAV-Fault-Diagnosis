# 鍦ㄧ嚎 A/B/C 瀹為獙鍛戒护

杩欎唤鏂囦欢鎸夊綋鍓嶆柊鐨勫湪绾垮疄楠岀粨鏋勬暣鐞嗭細

- A灞傦細鍏堥獙璇佹槸鍚﹀紩鍏ュ皯閲?`Real`
- B灞傦細鍦?A 灞傝儨鍑虹殑鍗忚涓婃瘮杈?backbone
- C灞傦細鍦?B 灞傛渶浼樺熀绾夸箣涓婂仛鍒涙柊妯″瀷

褰撳墠榛樿鐩綍锛?
- 鏈満浠撳簱锛歚D:\UAV-Fault-Diagnosis`
- 鏈満鏁版嵁锛歚D:\UAV-Data`
- 鏈満鍦ㄧ嚎瀹為獙鏍圭洰褰曪細`D:\UAV-Data\online_restart_v1`
- AutoDL 浠撳簱锛歚/root/autodl-tmp/UAV-Fault-Diagnosis`
- AutoDL 鏁版嵁锛歚/root/autodl-tmp/UAV-Data`
- AutoDL 鍦ㄧ嚎瀹為獙鏍圭洰褰曪細`/root/autodl-tmp/online_restart_v1`

## 1. 鏈満鏋勫缓 A 灞?split

```powershell
cd D:\UAV-Fault-Diagnosis
.\apps\\dashboard\\backend\.venv\Scripts\python.exe .\src\data\build_online_a_layer_splits.py `
  --manifest D:\UAV-Data\manifest\manifest_v2.csv `
  --dev-splits D:\UAV-Data\splits_v2\splits_v2.json `
  --output-dir D:\UAV-Data\online_restart_v1\splits `
  --seed 42 `
  --real-train-ratio 0.20 `
  --real-val-ratio 0.10 `
  --real-holdout-ratio 0.70
```

鐢熸垚缁撴灉锛?
- `D:\UAV-Data\online_restart_v1\splits\splits_online_a_baseline_v1.json`
- `D:\UAV-Data\online_restart_v1\splits\splits_online_a_fewshot_v1.json`
- `D:\UAV-Data\online_restart_v1\splits\protocol_online_a_v1.json`

## 2. 鏈満鏋勫缓 A1 baseline cache

```powershell
cd D:\UAV-Fault-Diagnosis
.\apps\\dashboard\\backend\.venv\Scripts\python.exe .\src\data\build_online_cache.py `
  --csv-dir D:\UAV-Data\processed_SIL_50hz `
  --manifest-path D:\UAV-Data\manifest\manifest_v2.csv `
  --splits D:\UAV-Data\online_restart_v1\splits\splits_online_a_baseline_v1.json `
  --output D:\UAV-Data\online_restart_v1\cache\cache_online_a_baseline_v1 `
  --window-size 150 `
  --stride 10 `
  --target-hz 50 `
  --fault-threshold 0.5 `
  --max-windows-per-file 8
```

## 3. 鏈満鏋勫缓 A2 few-shot cache

```powershell
cd D:\UAV-Fault-Diagnosis
.\apps\\dashboard\\backend\.venv\Scripts\python.exe .\src\data\build_online_cache.py `
  --csv-dir D:\UAV-Data\processed_SIL_50hz `
  --manifest-path D:\UAV-Data\manifest\manifest_v2.csv `
  --splits D:\UAV-Data\online_restart_v1\splits\splits_online_a_fewshot_v1.json `
  --output D:\UAV-Data\online_restart_v1\cache\cache_online_a_fewshot_v1 `
  --window-size 150 `
  --stride 10 `
  --target-hz 50 `
  --fault-threshold 0.5 `
  --max-windows-per-file 8
```

## 4. 鏈満鎵撳寘 A 灞?cache

```powershell
Compress-Archive `
  -Path D:\UAV-Data\online_restart_v1\cache\cache_online_a_baseline_v1 `
  -DestinationPath D:\UAV-Data\online_restart_v1\cache_online_a_baseline_v1.zip `
  -Force

Compress-Archive `
  -Path D:\UAV-Data\online_restart_v1\cache\cache_online_a_fewshot_v1 `
  -DestinationPath D:\UAV-Data\online_restart_v1\cache_online_a_fewshot_v1.zip `
  -Force
```

## 5. AutoDL 鐜鍑嗗

```bash
mkdir -p /root/autodl-tmp/UAV-Data
mkdir -p /root/autodl-tmp/online_restart_v1

cd /root/autodl-tmp/UAV-Fault-Diagnosis
python -m pip install -r requirements.txt
```

## 6. AutoDL 瑙ｅ帇 A 灞?cache

鍋囪浣犲凡缁忔妸涓や釜 zip 涓婁紶鍒颁簡锛?
- `/root/autodl-tmp/UAV-Data/cache_online_a_baseline_v1.zip`
- `/root/autodl-tmp/UAV-Data/cache_online_a_fewshot_v1.zip`

```bash
cd /root/autodl-tmp/UAV-Data
unzip -oq cache_online_a_baseline_v1.zip
unzip -oq cache_online_a_fewshot_v1.zip
```

## 7. AutoDL 璁粌 A1

```bash
cd /root/autodl-tmp/UAV-Fault-Diagnosis
python src/assets/models/train_online_detector.py \
  --model cnn1d \
  --mode baseline \
  --cache-dir /root/autodl-tmp/UAV-Data/cache_online_a_baseline_v1 \
  --output-dir /root/autodl-tmp/online_restart_v1/results/A1_online_cnn1d_baseline_v1 \
  --epochs 100 \
  --batch-size 512 \
  --lr 3e-4 \
  --hidden-size 64 \
  --dropout 0.2
```

## 8. AutoDL 璁粌 A2

```bash
cd /root/autodl-tmp/UAV-Fault-Diagnosis
python src/assets/models/train_online_detector.py \
  --model cnn1d \
  --mode fewshot \
  --cache-dir /root/autodl-tmp/UAV-Data/cache_online_a_fewshot_v1 \
  --output-dir /root/autodl-tmp/online_restart_v1/results/A2_online_cnn1d_fewshot_v1 \
  --epochs 100 \
  --batch-size 512 \
  --lr 3e-4 \
  --hidden-size 64 \
  --dropout 0.2
```

## 9. 鏈満姹囨€?A 灞傜粨璁?
鍏堟妸 AutoDL 涓婅繖涓や釜缁撴灉鐩綍鎷夊洖鏈満锛?
- `A1_online_cnn1d_baseline_v1`
- `A2_online_cnn1d_fewshot_v1`

鏀惧埌锛?
- `D:\UAV-Data\online_restart_v1\results\`

鐒跺悗鏈満鎵ц锛?
```powershell
cd D:\UAV-Fault-Diagnosis
.\apps\\dashboard\\backend\.venv\Scripts\python.exe .\scripts\compare_online_a_layer.py `
  --results-root D:\UAV-Data\online_restart_v1\results `
  --output-dir D:\UAV-Data\online_restart_v1\summary
```

杈撳嚭锛?
- `D:\UAV-Data\online_restart_v1\summary\online_a_layer_summary.csv`
- `D:\UAV-Data\online_restart_v1\summary\online_a_layer_summary.md`

## 10. 濡傛灉 A2 鑳滃嚭锛岃繘鍏?B 灞?
濡傛灉 A 灞傜粨璁烘槸 `few-shot Real 鎴愮珛`锛孊 灞傜粺涓€浣跨敤锛?
- `cache_online_a_fewshot_v1`

AutoDL 鎵归噺璁粌鍛戒护锛?
```bash
cd /root/autodl-tmp/UAV-Fault-Diagnosis
python scripts/run_online_backbone_suite.py \
  --python-bin python \
  --mode fewshot \
  --cache-dir /root/autodl-tmp/UAV-Data/cache_online_a_fewshot_v1 \
  --output-root /root/autodl-tmp/online_restart_v1/b_layer_models \
  --summary-dir /root/autodl-tmp/online_restart_v1/b_layer_summary \
  --models cnn1d bilstm transformer mstca \
  --epochs 100 \
  --batch-size 512 \
  --lr 3e-4 \
  --hidden-size 64 \
  --dropout 0.2 \
  --transformer-layers 2
```

濡傛灉 A1 鑳滃嚭锛屾妸涓婇潰鐨?`cache-dir` 鏀规垚锛?
- `/root/autodl-tmp/UAV-Data/cache_online_a_baseline_v1`

骞舵妸 `--mode fewshot` 鏀规垚锛?
- `--mode baseline`

## 11. B 灞傛眹鎬荤粨鏋?
```bash
cd /root/autodl-tmp/online_restart_v1/b_layer_summary
ls
cat online_backbone_ranking.md
```

鎶婁笅闈㈢洰褰曟媺鍥炴湰鏈猴細

- `/root/autodl-tmp/online_restart_v1/b_layer_summary`
- B 灞傚啝鍐涙ā鍨嬬洰褰?
## 12. C 灞傝缁冨垱鏂版ā鍨?
濡傛灉 B 灞傚啝鍐涘凡缁忔槸 `mstca`锛岄偅 C 灞傚氨璁粌 `kg_mstca` 鐩存帴瀵规墦銆?
濡傛灉 B 灞傚啝鍐涗笉鏄?`mstca`锛孋 灞備篃杩樻槸璁粌 `kg_mstca`锛屾渶鍚庢嬁瀹冨拰 B 灞傚啝鍐涙瘮杈冦€?
AutoDL 鍛戒护锛?
```bash
cd /root/autodl-tmp/UAV-Fault-Diagnosis
python src/assets/models/train_online_detector.py \
  --model kg_mstca \
  --mode fewshot \
  --cache-dir /root/autodl-tmp/UAV-Data/cache_online_a_fewshot_v1 \
  --output-dir /root/autodl-tmp/online_restart_v1/results/C1_online_kg_mstca_v1 \
  --epochs 100 \
  --batch-size 512 \
  --lr 3e-4 \
  --hidden-size 64 \
  --dropout 0.2
```

濡傛灉鏈€缁堝崗璁笉鏄?`fewshot`锛屾妸锛?
- `--mode fewshot`
- `cache_online_a_fewshot_v1`

鏇挎崲鎴?baseline 瀵瑰簲鍊煎嵆鍙€?
## 13. 鏈€寤鸿浼樺厛璺戠殑椤哄簭

鏃堕棿绱ф椂寤鸿鎸夎繖涓『搴忥細

1. 鏈満鏋勫缓 `A1 / A2` 涓ゅ cache
2. AutoDL 璺?`A1 / A2`
3. 鍏堢‘瀹氬埌搴曠敤 `baseline` 杩樻槸 `fewshot`
4. 鍐嶈窇 B 灞?`cnn1d / bilstm / transformer / mstca`
5. 鏈€鍚庤窇 C 灞?`kg_mstca`

涓嶈涓€寮€濮嬪氨鐩存帴鎶?`kg_mstca` 鍜屾墍鏈?backbone 涓€璧锋贩璺戯紝涓嶇劧瀹為獙缁撹浼氫贡銆?