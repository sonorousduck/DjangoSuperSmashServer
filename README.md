# Todo
### Post Request
    - This will be taking all of the gamestate data, introducing it into the Agents' memory (Either player one or player two). 
    - Then, it will take that gamedata, and call an async training function
    - This will create the bestAgent model, that the get request will be looking for. (Thus, if file does not exist, create it upon model creation)


### Get Request
    - This will return the weights of the model, as it is currently. (Even if training hasn't completed yet)
    - This will also handle the logic of determining the best AI out of the two, by at least 5%, and using that one. (If conditions not fulfilled, keep current weights)
    - 

