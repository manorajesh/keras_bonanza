from tensorflow import keras
model = keras.models.load_model("models/prime_model.v1.h5")

usr_input = ""
while True:
    usr_input = input("Enter a number to predict: ")
    if usr_input == "exit":
        break
    try:
        usr_input = int(usr_input)
    except:
        print("Please enter a valid number.")
        continue
    manual_prediction = model.predict([usr_input])
    prediction = "prime" if round(manual_prediction[0][0]) == 0  else "not prime"
    print(prediction)