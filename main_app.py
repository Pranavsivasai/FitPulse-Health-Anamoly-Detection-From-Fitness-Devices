import streamlit as st

def main():
    st.title("My First Streamlit App")
    st.write("Hi Students")

    # Step 2 - Text Input
    name = st.text_input("Enter your name:")

    if name:
        st.write("You typed:", name)

    # Step 3 - Greet Button
    if st.button("Greet"):
        st.write("Hello", name)

    st.write("---")

    # Step 5 - Calculator
    st.subheader("Simple Calculator")

    a = st.number_input("First number:", value=0)
    b = st.number_input("Second number:", value=0)

    operation = st.selectbox(
        "Choose operation:",
        ["Add", "Subtract", "Multiply"]
    )

    if st.button("Calculate"):

        if operation == "Add":
            result = a + b

        elif operation == "Subtract":
            result = a - b

        elif operation == "Multiply":
            result = a * b

        st.write("Hello", name)
        st.write("Operation:", operation)
        st.write("Result:", result)


if __name__ == "__main__":
    main()
