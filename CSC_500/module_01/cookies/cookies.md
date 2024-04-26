Chocolate Chip Pumpkin Cookes

START

INPUT: cookies_per_batch = 18

DISPLAY: How many cookies would you like?

INPUT: num_cookies = input() 

PROCESS: num_batches = math.ciel(num_cookies / cookies_per_batch)

DECISION: if num_batches > 0:
            continue
          else:
            goto END

PROCESS: Preheat oven to 350 degrees fahrenheit.

INPUT: ingredients = {
    "pumpkin_puree" : 1 cup,
    "white_sugar" : 0.5 cup,
    "brown_sugar" : 0.5 cup,
    "vanilla_extract" : 1 tsp,
    "eggs" : 1 egg,
    "flour" : 1 cup,
    "baking_powder" : 2 tsp,
    "salt" : 0.5 tsp,
    "cinnamon" : 0.125 tsp,
    "chocolate_chips" : 1 cup
}

DECISION: if num_batches > 1:
----PROCESS: for each ingredient in ingredients:
                ingredient *= num_batches

PROCESS: In first mixing bowl, mix pumpkin, white sugar, brown sugar, vanilla, and the egg(s).

PROCESS: In second mixing bowl, mix flour, baking powder, salt, and cinnamon.

PROCESS: Pour second mixing bowl into first mixing bowl and mix together.

PROCESS: Mix chocolate chips into the first mixing bowl with all the previously mixed ingredients.

PROCESS[0]: Line baking sheet with parchment paper and spray with nonstick spray.

PROCESS[1]: Scoop cookie batter onto prepared baking sheet(Assume baking sheet fits one batch of 18 cookies).

PROCESS[2]: Place loaded baking sheet into oven and bake at 350 degrees fahrenheight for 12 minutes. 

PRECESS[3]: Place cookies on plate and let cool.

DECISION: If num_batches > 1: 
            repeat PROCESS[0:3] num_batches-1 times
          else:
            continue

DISPLAY: Would you like to make more cookies?

INPUT: more_cookies = input()

DECISION: if more_cookies: 
            repeat all
          else:
            continue

PROCESS: Turn off oven and clean up.

END

