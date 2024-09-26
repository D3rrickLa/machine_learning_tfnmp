let mediaRecoder; 
let ws; 
let stream;


function startStreaming() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error("Media devices API is not supported");
        return;
    }

    navigator.mediaDevices.getUserMedia({video: {
            width: { ideal: 640 }, 
            height: { ideal: 480 },
            frameRate: { ideal: 30 }
        }
    })
    .then(stream => {
        const video = document.getElementById("video")
        video.srcObject = stream; 

        ws = new WebSocket("ws://localhost:8001/ws");
        mediaRecoder = new MediaRecorder(stream);
        ws.onopen = () => {
            console.log("WebSocket connection opened."); 
            
            mediaRecoder.ondataavailable = async function(event) {
                if (event.data.size > 0 && ws && ws.readyState == WebSocket.OPEN) {                    
                    const canvas = document.createElement("canvas")
                    canvas.width = video.videoWidth || 640; // Set default width if videoWidth is not available
                    canvas.height = video.videoHeight || 480; // Set default height if videoHeight is not available
                    
                    var ctx = canvas.getContext('2d', { willReadFrequently: true });
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
                    var pixelData = ctx.getImageData(0,0, canvas.width, canvas.height).data
                    
                    var length = pixelData.length

                    var rawData = new ArrayBuffer(length / 4 * 3); // Removes the alpha component
                    var uint8View = new Uint8Array(rawData)
                  
                    // copy RGB buffer content to the new arraybuffer
                    for(var i = 0, j = 0; i< length; i+=4, j +=3) {
                        uint8View[j] = pixelData[i];         // Red
                        uint8View[j + 1] = pixelData[i + 1]; // Green
                        uint8View[j + 2] = pixelData[i + 2]; // Blue
                    }    
                    
                    ws.send(uint8View)
                    
                }
            };
            
            mediaRecoder.start(33.33);
            
        };
        
        ws.onmessage = (event) => {
            console.log("Message from server:", event.data)
        }

        ws.onclose = () => {
            console.log("WebSocket connection closed.")
        };

        ws.onerror = (error) => {
            console.error("WebSocket error:", error);
        };
  
    })
    .catch(error => {
        console.error("error accessing media devices", error);
    })
}

function stopSignal() {
    // fetch("http://localhost:8000/stop_processing", {
    //     method: "POST",
    //     headers: {
    //         "Content-Type": "application/json"
    //     },
    //     body: JSON.stringify({command: "stop"})
    // })
    // .then(response => response.json())
    // .then(data => console.log(`Stop signal response: ${data.message}`))
    // .catch(error => console.error(`Error sending stop signal: ${error}`));

    ws.close()
}

function stopStreaming() {
    
    if(mediaRecoder) {
        mediaRecoder.stop();
        mediaRecoder = null; 
    }

    if (ws) {
        stopSignal();
        ws.close();
        ws = null;
    }

    if (stream) { 
        let tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        stream = null;
    }
}

document.querySelector("#stopButton").disabled = true
document.querySelector("#startButton").addEventListener("click", startCapture)
document.querySelector("#stopButton").addEventListener("click", endCapture)


function startCapture() {
    document.querySelector("#startButton").disabled = true
    document.querySelector("#stopButton").disabled = false
    startStreaming()
}

function endCapture() {
    document.querySelector("#startButton").disabled = false
    document.querySelector("#stopButton").disabled = true
    stopStreaming()
}