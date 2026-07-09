const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const messages = document.getElementById("messages");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  appendMessage("user", question);
  input.value = "";

  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  const data = await response.json();

  if (!response.ok) {
    appendMessage("assistant", data.error || "Something went wrong.");
    return;
  }
  appendMessage("assistant", data.answer, data.citations);
});

function appendMessage(role, text, citations) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;
  wrapper.textContent = text;

  if (citations && citations.length) {
    const list = document.createElement("ul");
    list.className = "citations";
    citations.forEach((citation) => {
      const item = document.createElement("li");
      item.textContent = `${citation.title}: "${citation.snippet}"`;
      list.appendChild(item);
    });
    wrapper.appendChild(list);
  }

  messages.appendChild(wrapper);
  messages.scrollTop = messages.scrollHeight;
}
