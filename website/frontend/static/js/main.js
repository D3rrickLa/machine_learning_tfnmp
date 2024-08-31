let mediaRecoder; 
let ws; 
let stream;



function startStreaming() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error("Media devices API is not supported");
        return;
    }

    navigator.mediaDevices.getUserMedia({video: true})
    .then(stream => {
        const video = document.getElementById("video")
        video.srcObject = stream; 
        
        
        
        ws = new WebSocket("ws://localhost:8000/ws2");
        
        ws.onopen = () => {
            console.log("WebSocket connection opened."); 
            mediaRecoder = new MediaRecorder(stream);
            
            mediaRecoder.ondataavailable = function(event) {
                if (event.data.size > 0 && ws && ws.readyState == WebSocket.OPEN) {                    
                    var canvas = document.getElementById("canvas")
                    canvas.width = video.videoWidth || 640; // Set default width if videoWidth is not available
                    canvas.height = video.videoHeight || 480; // Set default height if videoHeight is not available
                    var ctx = canvas.getContext('2d', { willReadFrequently: true });
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
                    var pixelData = ctx.getImageData(0,0, canvas.width, canvas.height).data
                    
                    var length = pixelData.length
                    var buffer=  pixelData

                    var rawData = new ArrayBuffer(length / 4 * 3); // why 4 * 3??
                    var uint8View = new Uint8Array(rawData)
                
                    // copy RGB buffer content to the new arraybuffer
                    for(var i = 0, j = 0; i< length; i+=4, j +=3) {
                        uint8View[j] = buffer[i];         // Red
                        uint8View[j + 1] = buffer[i + 1]; // Green
                        uint8View[j + 2] = buffer[i + 2]; // Blue
                    }    
                    
                    ws.send(uint8View);
                }
            };
            
            mediaRecoder.start(66.66)
            
        };

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

function stopStreaming() {
    if(mediaRecoder) {
        mediaRecoder.stop();
        mediaRecoder = null; 
    }

    if (ws) {
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