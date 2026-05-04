# Dashboard Frontend

Vue 3 + Vite + Element Plus 鍓嶇銆?
褰撳墠椤甸潰妯″潡鍖呮嫭锛?
- 鐧诲綍
- 鎬昏
- 瀹炴椂鐩戞祴
- 璇婃柇涓績
- 鏃ュ織璧勪骇
- 璇婃柇鎶ュ憡
- AI 鍔╂墜

## 瀹夎渚濊禆

```powershell
cd D:\UAV-Fault-Diagnosis\apps\\dashboard\\frontend
npm install
```

## 鍚姩寮€鍙戠幆澧?
浠庝粨搴撴牴鐩綍鍚姩锛?
```powershell
cd D:\UAV-Fault-Diagnosis
powershell -ExecutionPolicy Bypass -File .\scripts\start_dashboard_frontend.ps1
```

濡傛灉宸茬粡杩涘叆鍓嶇鐩綍锛屼篃鍙互鐩存帴杩愯锛?
```powershell
cd D:\UAV-Fault-Diagnosis\apps\\dashboard\\frontend
npm run frontend
```

榛樿绔彛锛?
- `http://localhost:5173`

榛樿浼氬皢 `/api` 鍜?`/health` 浠ｇ悊鍒帮細

- `http://localhost:5000`

涔熷彲浠ラ€氳繃鐜鍙橀噺瑕嗙洊锛?
- `VITE_BACKEND_TARGET`

## 甯哥敤鍛戒护

```powershell
npm run frontend
npm run test
npm run build
```

## 褰撳墠鍓嶇缁撴瀯

```text
src/
鈹溾攢 app/
鈹溾攢 layouts/
鈹溾攢 modules/
鈹? 鈹溾攢 ai/
鈹? 鈹溾攢 assets/
鈹? 鈹溾攢 auth/
鈹? 鈹溾攢 diagnosis/
鈹? 鈹溾攢 monitoring/
鈹? 鈹溾攢 overview/
鈹? 鈹斺攢 reports/
鈹溾攢 shared/
鈹斺攢 styles/
```
