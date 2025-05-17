# Function to check if an object is a matrix
is_matrix_r <- function(obj) {
  is.matrix(obj)
}


variables <- list(
  "X" = X,
  "Y" = Y,
  "Z" = Z,
  "X2" = X2,
  "X_test" = X_test,
  "X2_test" = X2_test
)

for (name in names(variables)) {
  value <- variables[[name]]
  if (is_matrix_r(value)) {
    cat(paste(name, "is a matrix.\n"))
  } else {
    cat(paste(name, "is NOT a matrix.\n"))
  }
}