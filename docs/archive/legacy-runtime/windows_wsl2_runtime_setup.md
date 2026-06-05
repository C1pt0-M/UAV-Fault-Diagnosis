# Windows / WSL2 杩愯鏃惰縼绉昏鏄?
杩欎唤鏂囨。鍙鐩栧綋鍓嶇郴缁熺殑杩愯鏃惰縼绉伙紝涓嶆秹鍙婇噸鏂拌缁冩ā鍨嬨€?
## 1. 鎺ㄨ崘褰㈡€?
鎺ㄨ崘浣跨敤锛?
- Windows 瀹夸富鏈鸿礋璐ｇ紪杈戝櫒銆佹祻瑙堝櫒銆佹枃浠剁鐞?- WSL2 璐熻矗 Dashboard backend / frontend銆丳X4銆乼elemetry銆乥ash 鑴氭湰

鍘熷洜锛?
- 褰撳墠浠撳簱閲岀殑鍚姩鑴氭湰鍜?PX4 閾捐矾浠嶇劧鏄?Linux first
- `telemetry_manager`銆乣start_dashboard*.sh`銆乣process_rflymad_v2.sh` 鏇撮€傚悎鐩存帴璺戝湪 WSL2
- Windows 鍘熺敓鐜板湪鍙互璺戝悗绔?鍓嶇锛屼絾涓嶅缓璁妸 PX4/SITL 褰撲綔涓昏矾寰?
## 2. 杩愯鍓嶅噯澶?
### 2.1 浠撳簱浣嶇疆

浼樺厛鎶婁粨搴撴斁杩?WSL2 Linux 鏂囦欢绯荤粺锛屼緥濡傦細

```bash
/home/<user>/UAV-Fault-Diagnosis
```

涓嶆帹鑽愰暱鏈熸斁鍦?`/mnt/d/...`锛屽挨鍏舵槸娑夊強澶ч噺灏忔枃浠跺拰閫掑綊澶勭悊鏃躲€?
### 2.2 Python / Node

鍦?WSL2 鍐呭噯澶囷細

```bash
cd ~/UAV-Fault-Diagnosis/apps/dashboard/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd ../frontend
npm install
```

### 2.3 鍚庣閰嶇疆

鍦?`apps/dashboard/backend` 鍐咃細

```bash
cp .env.example .env
```

閲嶇偣鍙橀噺锛?
- `ML_PYTHON_BIN=`锛氱暀绌烘椂榛樿浣跨敤褰撳墠鍚庣瑙ｉ噴鍣?- `TELEMETRY_PYTHON_BIN=`锛氱暀绌烘椂榛樿澶嶇敤 `ML_PYTHON_BIN`
- `PX4_ROOT=~/PX4-Autopilot`
- `MODEL_ROOT=models`
- `OFFLINE_NORM_STATS=data/processed/seq_cache/norm_stats.json`
- `AI_KNOWLEDGE_DIR=knowledge_base`
- `ULG_PREPARE_SCRIPT=`锛氬鏋?`ULG -> CSV` 浠嶄緷璧栧閮ㄨ剼鏈紝寤鸿鐩存帴濉粷瀵硅矾寰?- `FAULT_DATASET_ROOT=`锛氬彧鏈夋病閰?`ULG_PREPARE_SCRIPT` 鏃舵墠闇€瑕?
## 3. 鍚姩鏂瑰紡

### 3.1 鍦?WSL2 鍐呯洿鎺ュ惎鍔?
```bash
cd ~/UAV-Fault-Diagnosis
bash scripts/start_dashboard.sh
```

鎴栧垎鍒惎鍔細

```bash
bash scripts/start_dashboard_backend.sh
bash scripts/start_dashboard_frontend.sh
```

### 3.2 鍦?Windows 瀹夸富妗ユ帴鍚姩 WSL2

浠撳簱鏍圭洰褰曟柊澧炰簡锛?
```text
scripts/start_dashboard_wsl.ps1
```

榛樿琛屼负锛?
- 鑷姩鎶婂綋鍓?Windows 浠撳簱璺緞杞崲鎴?WSL 璺緞
- 璋冪敤 `wsl.exe`
- 鍦?WSL 涓墽琛?`bash scripts/start_dashboard.sh`

浣跨敤鏂规硶锛?
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_dashboard_wsl.ps1
```

鍙€夌幆澧冨彉閲忥細

- `UAV_WSL_REPO`锛氭樉寮忔寚瀹?WSL 鍐呬粨搴撹矾寰勶紝渚嬪 `/home/jack/UAV-Fault-Diagnosis`
- `UAV_WSL_DISTRO`锛氭寚瀹氬彂琛岀増鍚嶇О

## 4. 褰撳墠宸插畬鎴愮殑杩佺Щ鐐?
杩欒疆宸茬粡瀹屾垚锛?
- `apps/dashboard/backend/config.py`
  - 鏀寔绌虹幆澧冨彉閲忓洖閫€
  - 鏀寔 repo-relative 璺緞瑙ｆ瀽
  - 鍘绘帀 `/home/jack/...` 榛樿鍊?- `apps/dashboard/backend/app/services/bridge.py`
  - `ULG_PREPARE_SCRIPT` 鏄惧紡鑴氭湰浼樺厛
  - `FAULT_DATASET_ROOT` 浠呬綔鍏滃簳
  - 缂哄け鏃舵姏鍑烘槑纭厤缃彁绀?- `apps/dashboard/backend/app/services/telemetry_manager.py`
  - 鍘绘帀鏃?Linux Python fallback
  - `PX4_ROOT` 鍙粠鍚庣閰嶇疆瑕嗙洊
  - 瀛愯繘绋嬩細璇濇帶鍒舵敼鎴愯法骞冲彴瀹夊叏鍐欐硶
- `scripts/start_dashboard_backend.sh`
  - 鏀寔 `.venv/bin/python`銆乣.venv/Scripts/python.exe`銆乣python3`銆乣python`
- `scripts/start_dashboard_frontend.sh`
  - 澧炲姞 `npm` 瀛樺湪鎬ф鏌?
## 5. 褰撳墠鏈鐞嗙殑閮ㄥ垎

杩欒疆鏄庣‘娌℃湁鍋氾細

- 閲嶈缁冩ā鍨?- 閲嶅缓 `RflyMAD` baseline
- 鍘熺敓 Windows first-class PX4/SITL 鏂规
- 鑷姩瀹夎 WSL 鍙戣鐗?/ 鑷姩瀹夎渚濊禆

## 6. 楠岃瘉寤鸿

浼樺厛鍋氳繖鍑犱釜 smoke test锛?
1. `apps/dashboard/backend/.env` 鏄惁宸茬粡鍘绘帀鏃?`/home/jack/...`
2. `bash scripts/start_dashboard_backend.sh` 鑳藉惁鎵惧埌姝ｇ‘ Python
3. `bash scripts/start_dashboard_frontend.sh` 鑳藉惁鎵惧埌 `npm`
4. Dashboard 鏄惁鑳芥甯稿垪鍑?`assets/models/`
5. 濡傛灉闇€瑕?PX4锛宍PX4_ROOT` 鏄惁鎸囧悜 WSL2 鍐呮湁鏁堢洰褰?
濡傛灉鍙槸鍏堟紨绀虹绾胯瘖鏂細

- 涓嶉渶瑕佸厛閰嶅ソ `PX4_ROOT`
- 涓嶉渶瑕佸厛閰嶅ソ `TELEMETRY_PYTHON_BIN`
- 鍙鍚庣 Python 鑳藉姞杞芥ā鍨嬨€佸墠绔兘鍚姩鍗冲彲
