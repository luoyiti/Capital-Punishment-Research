library(FSelector)

info <- data.frame(fruits = c("watermelon", "apple", "banana", "grape", "grapefruit", "lemon"))
info$sizes <- c("big", "medium", "medium","small" ,"medium", "small")
info$colors <- c("green", "red", "yellow", "green", "yellow", "yellow")
info$shapes <- c("round", "round", "thin", "round", "round", "round")

# get information gain results
information.gain(formula(info), info)
