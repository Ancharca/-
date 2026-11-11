import fuzzytool as fz
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

smell = fz.Variable("smell", (0, 100), terms=["good", "not bad", "horrible"])
shell_life = fz.Variable("shell_life", (0, 100), terms=["today", "resently", "long time"])
solution = fz.Variable("solution", (0, 100), terms=["leave", "worth eat", "probably not worth eat", "garbage"])

smell["good"] = fz.tri(0, 0, 50)
smell["not bad"] = fz.tri(25, 50, 75)
smell["horrible"] = fz.tri(50, 100, 100)

shell_life["today"] = fz.tri(0, 0, 50)
shell_life["recently"] = fz.tri(25, 50, 75)
shell_life["long time"] = fz.tri(50, 100, 100)

solution["leave"] = fz.tri(0, 0, 30)
solution["worth eat"] = fz.tri(20, 40, 60)
solution["probably not worth eat"] = fz.tri(40, 60, 80)
solution["garbage"] = fz.tri(70, 100, 100)

sys = fz.Mamdani(defuzz="centroid")

sys.rule(smell["good"] & shell_life["today"], solution["leave"])
sys.rule(smell["good"] & shell_life["recently"], solution["worth eat"])
sys.rule(smell["good"] & shell_life["long time"], solution["probably not worth eat"])

sys.rule(smell["not bad"] & shell_life["today"], solution["worth eat"])
sys.rule(smell["not bad"] & shell_life["recently"], solution["probably not worth eat"])
sys.rule(smell["not bad"] & shell_life["long time"], solution["garbage"])

sys.rule(smell["horrible"] & shell_life["today"], solution["probably not worth eat"])
sys.rule(smell["horrible"] & shell_life["recently"], solution["garbage"])
sys.rule(smell["horrible"] & shell_life["long time"], solution["garbage"])

st.title("Система нечеткого вывода по степени свежести продукта")

def trimf(x, a, b, c):
    y = np.zeros_like(x, dtype=float)
    left = (x >= a) & (x <= b)
    right = (x > b) & (x <= c)
    if b > a:
        y[left] = (x[left] - a) / (b - a)
    if c > b:
        y[right] = (c - x[right]) / (c - b)
    return y

smell_terms = {
    "good": (0, 0, 50),
    "not bad": (25, 50, 75),
    "horrible": (50, 100, 100),
}
shell_life_terms = {
    "today": (0, 0, 50),
    "recently": (25, 50, 75),
    "long time": (50, 100, 100),
}
solution_terms = {
    "leave": (0, 0, 30),
    "worth eat": (20, 40, 60),
    "probably not worth eat": (40, 60, 80),
    "garbage": (70, 100, 100),
}

x = np.linspace(0, 100, 501)

smell_val = st.slider("Запах (0 = свежий, 100 = ужасный)", 0, 100, 30, 1)
shell_val = st.slider("Как давно куплен (0 = сегодня, 100 = очень давно)", 0, 100, 40, 1)

result = sys(smell=smell_val, shell_life=shell_val)

st.metric("Вероятность, что продукт стоит выбросить", f"{result:.1f}%")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Запах")
    fig, ax = plt.subplots(figsize=(6, 3))
    for name, (a, b, c) in smell_terms.items():
        ax.plot(x, trimf(x, a, b, c), label=name)
    ax.axvline(smell_val, color="black", linestyle="--", label=f"вход = {smell_val}")
    ax.set_xlabel("Запах (0–100)")
    ax.set_ylabel("p")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    st.pyplot(fig)

with col2:
    st.subheader("Срок хранения")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    for name, (a, b, c) in shell_life_terms.items():
        ax2.plot(x, trimf(x, a, b, c), label=name)
    ax2.axvline(shell_val, color="black", linestyle="--", label=f"вход = {shell_val}")
    ax2.set_xlabel("Давность покупки (0–100)")
    ax2.set_ylabel("p")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=8)
    st.pyplot(fig2)

st.subheader("Выход: вероятность выбросить")
fig3, ax3 = plt.subplots(figsize=(10, 3))
for name, (a, b, c) in solution_terms.items():
    ax3.plot(x, trimf(x, a, b, c), label=name)
ax3.axvline(result, color="red", linestyle="--", label=f"выход = {result:.1f}%")
ax3.set_xlabel("Вероятность выбросить, %")
ax3.set_ylabel("p")
ax3.set_ylim(-0.05, 1.05)
ax3.legend(fontsize=8)
st.pyplot(fig3)
