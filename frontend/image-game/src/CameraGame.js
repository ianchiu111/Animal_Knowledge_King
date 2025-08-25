import React, { useRef, useState } from "react";

function CameraGame() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [streaming, setStreaming] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // 啟動攝影機
  const startCamera = async () => {
    if (!streaming) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });  // 使用者要同意開啟鏡頭
        videoRef.current.srcObject = stream;
        setStreaming(true);
      } catch (err) {
        alert("無法啟動相機：" + err.message);
      }
    }
  };

  // 擷取一張圖片並送到 backend
  const captureAndSend = async () => {
    if (!videoRef.current) return;
    // 將 video 畫面畫到 canvas 上
    const width = videoRef.current.videoWidth;
    const height = videoRef.current.videoHeight;
    canvasRef.current.width = width;
    canvasRef.current.height = height;
    const ctx = canvasRef.current.getContext("2d");
    ctx.drawImage(videoRef.current, 0, 0, width, height);

    // 轉成 blob (你也可以用 toDataURL)
    canvasRef.current.toBlob(async (blob) => {
      if (!blob) return;
      setLoading(true);
      setResult(null);

      // 上傳到 backend 辨識
      const formData = new FormData();
      formData.append("file", blob, "capture.png");
      try {
        const res = await fetch("http://localhost:5000/api/predict", {
        method: "POST",
        body: formData,
        }); 
        const data = await res.json();
        setResult(data.label || "未知結果");
      } catch (e) {
        setResult("辨識失敗");
      } finally {
        setLoading(false);
      }
    }, "image/png");
  };

  // 關閉攝影機
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setStreaming(false);
    }
  };

  return (          // Return JSX 語法
    <div style={{textAlign: "center"}}>
      <h2>影像辨識小遊戲</h2>
      <video        // 顯示攝影機即時畫面
        ref={videoRef}
        style={{ width: 400, height: 300, border: "1px solid #ddd" }}
        autoPlay
        muted
      ></video>
      
      <canvas ref={canvasRef}></canvas> 
      <div style={{ margin: 20 }}>          
        <button onClick={startCamera} disabled={streaming}>啟動相機</button>
        <button onClick={captureAndSend} disabled={!streaming || loading}>拍攝</button>
        <button onClick={stopCamera} disabled={!streaming}>關閉相機</button>
      </div>
      {loading && <div>辨識中...</div>}
      {result && <div style={{fontSize: 24}}>辨識結果：{result}</div>}
    </div>
  );
}

export default CameraGame;
