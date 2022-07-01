
library(nloptr)

data <- df%>%
  rename_at(vars(ends_with(paste0("predictions_", level))), ~ "PRED")%>%
  rename_at(vars(ends_with(paste0("ground_truth_", level))), ~ "TRUTH")%>%
  mutate(Results = ground_truth_5 == predictions_5,
         Score = probabilities - probabilities_k2)%>%
  select(Results, Score, liasseNb, PRED, TRUTH)

N <- nrow(data)
q <- 0.1
level <- 1
Target2Revise <- floor(N*q)

ni <- data%>%
  group_by(TRUTH)%>%
  summarise(
    N = n()
  )%>%
  pull(N)

# Objective Function
eval_f <- function(x){
  
  Number2Remove <- round(x*ni)
  Liasse2Remove <- c()
  ListClasses <- data%>% pull(TRUTH)%>%unique()%>%sort()
  
  for (i in 1:length(ListClasses)) {
    idx2Remove <- data%>%
      subset(PRED == ListClasses[i])%>%
      arrange(desc(Score))%>%
      slice_tail(n = Number2Remove[i])%>%
      pull(liasseNb)
    Liasse2Remove <- c(Liasse2Remove, idx2Remove)
  }
  
  accuracy <- data %>%
    mutate(
      Results = case_when(
        liasseNb %in% Liasse2Remove ~ T,
        TRUE ~ Results
      ))%>%
    pull(Results)%>%
    mean()
  
  return (accuracy)
}
# essayer avec 19 classe dans la fonction pour se débarasser de la contrainte

# Equality constraints
eval_g_eq <- function(x){
  return (as.numeric(Target2Revise -  floor(x%*%ni)))
}

sample(100, 100)
# Lower and upper bounds
lb <- rep(0, length(ni))
ub <- rep(0.4, length(ni))

#initial values
x0 <- rep(0.1, length(ni)) 

# Set optimization options.
local_opts <- list( "algorithm" = "NLOPT_LD_MMA", "xtol_rel" = 1.0e-15 )
opts <- list( "algorithm"= "NLOPT_GN_CRS2_LM",
              "xtol_rel"= 1.0e-1,
              "maxeval"= 1600,
              "local_opts" = local_opts,
              "print_level" = 0 )

res <- nloptr ( x0 = x0,
                eval_f = eval_f,
                lb = lb,
                ub = ub,
                eval_g_eq = eval_g_eq,
                opts = opts
)


######### dd #####
eval_f <- function(x, data){
  
  Number2Remove <- round(x*ni)
  Liasse2Remove <- c()
  ListClasses <- data%>% pull(TRUTH)%>%unique()%>%sort()
  
  for (i in 1:length(ListClasses)) {
    idx2Remove <- data%>%
      subset(PRED == ListClasses[i])%>%
      arrange(desc(Score))%>%
      slice_tail(n = Number2Remove[i])%>%
      pull(liasseNb)
    Liasse2Remove <- c(Liasse2Remove, idx2Remove)
  }
  
  accuracy <- data %>%
    mutate(
      Results = case_when(
        liasseNb %in% Liasse2Remove ~ T,
        TRUE ~ Results
      ))%>%
    pull(Results)%>%
    mean()
  
  return (accuracy)
}

sample <- df%>%
  rename_at(vars(ends_with(paste0("predictions_", level))), ~ "PRED")%>%
  rename_at(vars(ends_with(paste0("ground_truth_", level))), ~ "TRUTH")%>%
  mutate(Results = ground_truth_5 == predictions_5,
         Score = probabilities - probabilities_k2)%>%
  select(Results, Score, liasseNb, PRED, TRUTH)

N <- nrow(sample)
q <- 0.1
level <- 1
Target2Revise <- floor(N*q)

ni <- data%>%
  group_by(TRUTH)%>%
  summarise(
    N = n()
  )%>%
  pull(N)

MatProp <- replicate(19,runif(10000,0,0.2))
Prop2Check <- as.data.frame((Target2Revise - MatProp %*% ni[2:length(ni)]) / ni[1])%>%
  bind_cols(data.frame(MatProp))%>%
  subset((V1 <= 0.4) & (V1 >= 0))


# Objective Function
store <- c()

start_time <- Sys.time()
for (i in 1:nrow(Prop2Check)) {
  
  store <- c(store,
             eval_f(Prop2Check %>%slice(i) %>%as.numeric(), 
                    sample)
  )
  cat("*** Iteration", i, "\n")
} 
end_time <- Sys.time()
end_time - start_time


x <- Prop2Check%>%
  mutate(Accuracy = store)%>%
  arrange(desc(Accuracy))%>%
  slice(1)%>%
  as.numeric()
