#Misc. stuff on loops and apply function

#Pre-allocate vectors prior to looping if using for loop to make memory more
#efficient

vector("list", length = 10)

#Then do for-loop

#vapply: like lapply, but for vectors with the extra FUN.VALUE argue

vapply(
  1:10,
  sqrt,
  numeric(1) #FUN.VALUE; tell vapply what you expect your value to look like
)

