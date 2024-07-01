# Simple Python code to print something interesting after user input
import random

def get_fun_fact(name):
    # List of fun facts
    fun_facts = [
        "did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible!",
        "there are more possible iterations of a game of chess than there are atoms in the known universe.",
        "a single strand of spaghetti is called a 'spaghetto'.",
        "the shortest war in history was between Britain and Zanzibar on August 27, 1896. Zanzibar surrendered after 38 minutes.",
        "a flock of crows is known as a murder."
    ]
    # Randomly select a fun fact
    fact = random.choice(fun_facts)
    return f"Hello {name}, {fact}"

# Get user input
name = input("Please enter your name: ")

# Print personalized message with a fun fact
print(get_fun_fact(name))

