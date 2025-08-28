import React, { useRef, useState } from "react";

function CameraGame() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [streaming, setStreaming] = useState(false);
  const [questionObj, setQuestionObj] = useState(null); // 問題資料
  const [label, setLabel] = useState(null);             // 辨識出來的動物
  const [loading, setLoading] = useState(false);
  const [selectedOption, setSelectedOption] = useState(null); // 使用者選擇
  const [result, setResult] = useState(null);           // 結果: "correct" / "wrong"
  const [showExplanation, setShowExplanation] = useState(false); // 是否顯示解釋
  const [photoUrl, setPhotoUrl] = useState(null); // 新增：拍照後預覽

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

    // 新增：把canvas畫面存成圖片URL預覽
    const dataUrl = canvasRef.current.toDataURL("image/png");
    setPhotoUrl(dataUrl);

    // 轉成 blob (你也可以用 toDataURL)
    canvasRef.current.toBlob(async (blob) => {
      if (!blob) return;
      setLoading(true);
      setResult(null);
      setShowExplanation(false);
      setSelectedOption(null);

      // 上傳到 backend 辨識
      const formData = new FormData();
      formData.append("file", blob, "capture.png");
      try {
        const res = await fetch("http://localhost:5000/api/v1/predict_image_category", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        setLabel(data.label || "未知");
        setQuestionObj(data.question); // 有題目的時候，才會渲染問答
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

  // 選擇答案後按送出
  const handleSubmit = () => {
    if (!questionObj || selectedOption === null) return;

    // 確認玩家的選擇與答案是否一致（這裡以 correct_answer 填序號為標準，1-based）
    if (selectedOption === parseInt(questionObj.correct_answer, 10) - 1) {
      setResult("correct");
    } else {
      setResult("wrong");
    }
    setShowExplanation(true);
  };

  // 過濾掉 null 的選項，這樣option4為null時自動不顯示（保留舊註解與新增說明）
  const options = questionObj ? questionObj.options.filter(opt => opt !== null) : [];

  return (
    <div style={{ width: "100vw", minHeight: "100vh", background: "#fff" }}>
      <h1 style={{ textAlign: "center", marginTop: 12 }}>Animal Knowledge Game</h1>
      {/* 主區塊改用 flex row 排版（左：鏡頭、右：問答） */}
      <div style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "flex-start",
        gap: 40,
        marginTop: 10
      }}>
        {/* ----------- 左側：鏡頭+操作+照片 ----------- */}
        <div style={{ minWidth: 430, display: "flex", flexDirection: "column", alignItems: "center" }}>
          <video
            ref={videoRef}
            style={{ width: 400, height: 300, border: "1px solid #ddd" }}
            autoPlay
            muted
          ></video>
          <canvas ref={canvasRef} style={{ display: "none" }}></canvas>
          <div style={{ margin: "18px 0" }}>
            <button onClick={startCamera} disabled={streaming}>啟動相機</button>
            <button onClick={captureAndSend} disabled={!streaming || loading} style={{ marginLeft: 8 }}>拍攝並進行問答</button>
            <button onClick={stopCamera} disabled={!streaming} style={{ marginLeft: 8 }}>關閉相機</button>
          </div>
          {/* 新增：拍照後照片預覽 */}
          {photoUrl &&
            <div>
              <div style={{ margin: "4px 0 3px 0", color: "#888", fontSize: 24 }}>拍攝的照片：</div>
              <img src={photoUrl} alt="拍攝照片" style={{ width: 360, border: "1px solid #eee", borderRadius: 8 }} />
            </div>
          }
          {/* 顯示辨識結果 */}
          {label && <div style={{ fontSize: 19, margin: "12px 0" }}>辨識結果：<b>{label}</b></div>}
          {loading && <div style={{ margin: 10, color: "#555" }}>辨識中...</div>}
        </div>

        {/* ----------- 右側：問答區 ----------- */}
        <div style={{ flex: 1, minWidth: 350, maxWidth: 600, marginLeft: 10 }}>
          {/* 如果後端有回題目，就顯示問答 */}
          {questionObj && (
            <div>
              <div style={{ fontSize: 26, marginBottom: 18, textAlign: "left", fontWeight: 500 }}>
                {questionObj.question}
              </div>
              <div style={{ textAlign: "left", marginBottom: 16 }}>
                {options.map((opt, idx) => (
                  <div key={idx} style={{ marginBottom: 7 }}>
                    <label style={{ fontSize: 22, display: "flex", alignItems: "center" }}>
                      <input
                        type="radio"
                        name="option"
                        checked={selectedOption === idx}
                        onChange={() => setSelectedOption(idx)}
                        disabled={result !== null} // 有結果時不能再選
                        style={{ marginRight: 10 }}
                      />
                      {opt}
                    </label>
                  </div>
                ))}
              </div>
              {/* 送出按鈕 */}
              {!result && (
                <button
                  onClick={handleSubmit}
                  disabled={selectedOption === null}
                  style={{ marginTop: 10, padding: "5px 26px", fontSize: 16 }}
                >
                  送出答案
                </button>
              )}
              {/* 答案結果判斷 */}
              {result === "correct" && (
                <div style={{ color: "green", fontSize: 20, marginTop: 18, fontWeight: 600 }}>
                  ✅ 恭喜你答對了！
                </div>
              )}
              {result === "wrong" && (
                <div style={{ color: "red", fontSize: 20, marginTop: 18, fontWeight: 600 }}>
                  ❌ 答錯囉！
                </div>
              )}
              {/* 顯示解釋 */}
              {showExplanation && (
                <div style={{ fontSize: 16, marginTop: 20, textAlign: "left" }}>
                  <b>解釋：</b>{questionObj.explanation}
                </div>
              )}
            </div>
          )}

          {/* 如果辨識有動物但沒有題目 */}
          {label && !questionObj && !loading && (
            <div style={{ color: "orange", fontSize: 18, marginTop: 36 }}>這個動物還沒有題目喔！</div>
          )}
        </div>
      </div>
    </div>
  );
}

export default CameraGame;
