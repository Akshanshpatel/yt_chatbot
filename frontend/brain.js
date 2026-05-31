const BACKEND_URL = "https://yt-chatbot-68zq.onrender.com/";

const loadBtn = document.getElementById("loadBtn");
const askBtn = document.getElementById("askBtn");

const videoUrlInput = document.getElementById("videoUrl");
const questionInput = document.getElementById("question");
const chatBox = document.getElementById("chatBox");

function addMessage(text, type) {

    const div = document.createElement("div");

    div.classList.add("message");
    div.classList.add(type);

    div.innerText = text;

    chatBox.appendChild(div);

    chatBox.scrollTop = chatBox.scrollHeight;
}

loadBtn.addEventListener("click", async () => {

    const url = videoUrlInput.value.trim();

    if (!url) {
        alert("Please enter a YouTube URL");
        return;
    }

    try {

        loadBtn.disabled = true;
        loadBtn.innerText = "Loading...";

        const response = await fetch(
            `${BACKEND_URL}/load-video`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    url: url
                })
            }
        );

        const data = await response.json();

        if (response.ok) {

            alert("Video loaded successfully!");

        } else {

            alert(data.detail);
        }

    } catch (error) {

        console.error(error);
        alert("Failed to load video.");

    } finally {

        loadBtn.disabled = false;
        loadBtn.innerText = "Load Video";
    }
});

askBtn.addEventListener("click", async () => {

    const question = questionInput.value.trim();

    if (!question) return;

    addMessage(question, "user");

    questionInput.value = "";

    try {

        const response = await fetch(
            `${BACKEND_URL}/ask`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    question: question
                })
            }
        );

        const data = await response.json();

        addMessage(data.answer, "bot");

    } catch (error) {

        console.error(error);

        addMessage(
            "Something went wrong.",
            "bot"
        );
    }
});