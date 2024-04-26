function get_food_cost()
    food_cost = input()
    return food_cost

function get_tip(food_cost)
    tip = food_cost * 0.18
    return tip

function get_tax(food_cost)
    tax = food_cost * 0.07
    return tax

function calculate_total(food_cost, tip, tax)
    total = food_cost + tip + tax
    return total

function main()
    food_cost = get_food_cost()
    tip = get_tip(food_cost)
    tax = get_tax(food_cost)
    total = calculate_total(food_cost, tip, tax)

    