document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    // For local development, the backend runs on http://localhost:8000
    // For production, this URL will need to be the deployed backend URL.
    // Production backend URL
    const backendUrl = 'https://ai-delivery-framework-gemini-production.up.railway.app/gemini/chat';
    // Example production URL: const backendUrl = 'https://your-productpod-backend.yourdomain.com/gemini/chat';

    let chatHistory = []; // To store messages {role: "user" | "model", parts: ["text"]}

    function appendMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(sender === 'user' ? 'user-message' : 'model-message');

        // Basic Markdown-like rendering for newlines
        messageDiv.innerHTML = text.replace(/\n/g, '<br>');

        chatWindow.appendChild(messageDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight; // Auto-scroll to bottom
    }

    async function sendMessage() {
        const messageText = userInput.value.trim();
        if (!messageText) return;

        appendMessage(messageText, 'user');
        chatHistory.push({ role: 'user', parts: [messageText] });
        userInput.value = ''; // Clear input field

        try {
            // Show a thinking indicator (optional)
            appendMessage("<i>Thinking...</i>", 'model');
            const thinkingMessage = chatWindow.lastChild;


            const response = await fetch(backendUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    messages: chatHistory
                    // current_tools_output is not sent from client in this version
                }),
            });

            if (thinkingMessage && thinkingMessage.parentNode === chatWindow) {
                chatWindow.removeChild(thinkingMessage); // Remove thinking message
            }

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: "Unknown error occurred" }));
                console.error('Error from backend:', response.status, errorData);
                appendMessage(`Error: ${errorData.detail || response.statusText}`, 'model');
                // Do not add error to chat history for Gemini to process
                return;
            }

            const responseData = await response.json();

            if (responseData.text) {
                appendMessage(responseData.text, 'model');
                chatHistory.push({ role: 'model', parts: [responseData.text] });
            } else if (responseData.function_call_suggestion) {
                // This might happen if the backend indicates a tool call is still pending after max iterations
                const suggestionText = `The model is suggesting to use the tool: ${responseData.function_call_suggestion}. Further interaction might be needed or the backend needs to resolve this.`;
                appendMessage(suggestionText, 'model');
                // Do not add this specific message to history as it's a meta-comment.
            } else {
                appendMessage("Received an empty response from the model.", 'model');
                 // Potentially add an empty model part to history if that's how you want to treat it
                // chatHistory.push({ role: 'model', parts: [""] });
            }

        } catch (error) {
            console.error('Failed to send message or process response:', error);
            const thinkingMessage = Array.from(chatWindow.childNodes).find(node => node.innerHTML === "<i>Thinking...</i>");
            if (thinkingMessage) {
                chatWindow.removeChild(thinkingMessage);
            }
            appendMessage(`Network or application error: ${error.message}`, 'model');
        }
    }

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    // Initial message or greeting (optional)
    // appendMessage("Hello! I'm ProductPod. How can I help you today?", 'model');
    // chatHistory.push({ role: 'model', parts: ["Hello! I'm ProductPod. How can I help you today?"] });
});
