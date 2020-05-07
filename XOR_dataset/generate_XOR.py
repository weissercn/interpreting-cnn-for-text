import numpy as np
import random

def generate(fname_tok, fname_cat, N_samples):
    f_tok = open(fname_tok, "w")
    f_cat = open(fname_cat, "w")


    filler_phrases = [
    "while on the other hand"
    ]

    preference_verbs = ["prefers", "likes"]

    days = ["Monday", "Tuesday"]

    A_names = ["Alica"] # we could also sample these names
    B_names = ["Bob"] 


    n = 0

    fraction_sameday = 0.6

    while True:
        if n >= (N_samples-1) : break
        #if (n%10)==0: print n

        # choice() 1
        # sample() many without replacement
        # choices() many with replacement

        A_name = random.choice(A_names) # could sample
        B_name = random.choice(B_names) # could sample
        A_preference_verb, B_preference_verb = random.choices(preference_verbs,k=2 )
        filler_phrase = random.choice(filler_phrases)

        bool_sameday = (np.random.random() < fraction_sameday)

        if bool_sameday: A_day = random.choice(days); B_day = A_day
        else: A_day, B_day = random.sample(days, 2)
            

        sentence = "{} {} {} {} {} {} {}\n".format(A_name, A_preference_verb, A_day, filler_phrase, B_name, B_preference_verb, B_day)
        f_tok.write(sentence)
        f_cat.write("{}\n".format(bool_sameday+1))
        n +=1

    f_tok.close()
    f_cat.close()


if __name__=="__main__":
    generate("train.txt.tok", "train.cat", 10000)
    generate("test.txt.tok", "test.cat", 10000)





