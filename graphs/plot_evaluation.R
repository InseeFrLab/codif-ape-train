rm(list = ls())
setwd("~/codification-ape/graphs")
library(aws.s3)
library(dplyr)
install.packages("tibble")
install.packages("caret")
library(caret)
source("theme_custom.R")
aws.s3::get_bucket("projet-ape", region = "", prefix = "data/preds_test.csv")


##### Data importation #####
df_test <- 
  aws.s3::s3read_using(
    FUN = readr::read_csv,
    # Mettre les options de FUN ici
    object = "/data/preds_test.csv",
    bucket = "projet-ape",
    opts = list("region" = "")
  )
df_test <- df_test %>%
  mutate(Type="Test")

df_gu <- 
  aws.s3::s3read_using(
    FUN = readr::read_csv,
    # Mettre les options de FUN ici
    object = "/data/preds_gu.csv",
    bucket = "projet-ape",
    opts = list("region" = "")
  )

df <- df_gu %>%
  mutate(Type="GU")%>%
  bind_rows(df_test)

latex_percent <- function(x){
  x <- plyr::round_any(x,scales:::precision(x)/100)
  stringr::str_c(x * 100, "\\%")
}

##### Matrice de confusion niveau 1 ##### 
PlotConfusionMatrix <- function(data, type){
  
  TotalSize <- data%>%subset(Type %in% type)%>%nrow()
  
  Shares <- data%>%
    subset(Type %in% type)%>%
    group_by(ground_truth_1, Type)%>%
    summarise(
      N = n(),
    )%>%
    mutate(Share = N/TotalSize*100)
  
  sample <- data%>%
    subset(Type %in% type)%>%
    mutate(ground_truth_1 = factor(ground_truth_1, levels = Shares$ground_truth_1),
           predictions_1 = factor(predictions_1, levels = Shares$ground_truth_1))%>%
    select(ground_truth_1, predictions_1)
  
  cm <- confusionMatrix(sample$ground_truth_1, sample$predictions_1)
  
  sample <- cm$table %>%
    data.frame() %>% 
    mutate(Prediction = factor(Prediction, levels = Shares$ground_truth_1)) %>%
    group_by(Reference) %>% 
    mutate(
      total = sum(Freq),
      perc = Freq / total*100
    )
  
  
  NewTitle <- paste0(sample$Reference, "\n(", rep(round(Shares$Share, 1), each=length(Shares$ground_truth_1)), "%)")
  names(NewTitle) <- sample$Reference 
  
  sample <- sample %>%
    mutate(Reference = recode(Reference, !!!NewTitle)
    )
  
  plot <-  ggplot(sample, aes(x=Reference, y= Prediction)) +
    geom_tile(aes(fill = perc))+
    scale_fill_gradient(low = "white", high = Palette_col[1]) +
    geom_text(aes(label = paste0(round(perc,0), "%")), size = 3)+
    theme_custom()+
    guides(fill = "none")  
  
  return(plot)
}
PlotConfusionMatrix(df, c("GU"))
PlotConfusionMatrix(df, c("Test"))

##### Aggrégation par moyenne ##### 
PlotAccuracies <- function(data, level, title){
  sample <- data %>%
    mutate(Results = .data[[paste0("predictions_", level)]] == .data[[paste0("ground_truth_", level)]]) %>%
    group_by(ground_truth_1, Type)%>%
    summarise(
      Accuracy = mean(Results)
    )
  
  plot <-ggplot(sample, aes(x=ground_truth_1, y=Accuracy, fill=Type))+
    ggtitle(title)+
    geom_bar(stat = "identity", position = position_dodge())+
    geom_text(aes(label=round(Accuracy,2)), position=position_dodge(width=0.9), vjust=-0.25)+
    scale_fill_manual(values=c(Palette_col))+
    theme_custom()
  return(plot)
}
PlotAccuracies(df, 1, "Accuracy au niveau 1")
PlotAccuracies(df, 5, "Accuracy au niveau 5 (aggrégation par moyenne)")

##### Aggrégation par moyenne avec proportion ##### 
PlotAccuracies_with_Shares <- function(data, type, level, title){
  
  TotalSize <- data%>%subset(Type %in% type)%>%nrow()
  sample <- data %>%
    subset(Type %in% type)%>%
    mutate(Results = .data[[paste0("predictions_", level)]] == .data[[paste0("ground_truth_", level)]]) %>%
    group_by(ground_truth_1, Type)%>%
    summarise(
      Accuracy = mean(Results),
      N = n(),
    )%>%
    mutate(Share = N/TotalSize*100,
           NewTitle = paste0(ground_truth_1, "\n(", round(Share, 1), "%)"))
plot <-ggplot(sample, aes(x=NewTitle, y=Accuracy, fill=Type))+
  ggtitle(title)+
  geom_bar(stat = "identity", position = position_dodge())+
  geom_text(aes(label=round(Accuracy,2)), position=position_dodge(width=0.9), vjust=-0.25)+
  scale_fill_manual(values=c(Palette_col))+
  theme_custom()
return(plot)
}
PlotAccuracies_with_Shares(df, "Test", 1, "Accuracy au niveau 1")
PlotAccuracies_with_Shares(df, "GU", 1, "Accuracy au niveau 1")
PlotAccuracies_with_Shares(df, "Test", 5, "Accuracy au niveau 5 (aggrégation par moyenne)")
PlotAccuracies_with_Shares(df, "GU", 5, "Accuracy au niveau 5 (aggrégation par moyenne)")

##### Construction de la base avec reprise des données ##### 
acc_w_rep <- function(data, q, type, level){

  accuracy <- data %>% 
    subset(Type %in% type)%>%
    select(probabilities, probabilities_k2, ends_with(paste0("_", level)))%>%
    rename_with(~ gsub(paste0("_", level,"$"), "", .x))%>%
    mutate(Score = probabilities - probabilities_k2,
           Results = ground_truth == predictions,
           Revisions = case_when(Score >= quantile(Score, q) ~ Results,
                                 TRUE ~ TRUE)
    )%>%
    pull(Revisions)%>%
    mean
  return(accuracy)
}

df_reprise <- tibble(Rate = character(), Level = character(), Type = character(), Accuracy = numeric())
for (type in c("GU", "Test")) {
  for (level in paste(1:5)) {
    for (rate in paste(seq(0,0.25,0.05))) {
      df_reprise <- df_reprise %>%
                        add_row(Rate = rate, 
                                Level = level,
                                Type = type,
                                Accuracy = acc_w_rep(df, 
                                                     as.double(rate), 
                                                     type, 
                                                     as.double(level)
                                                    )
                                )
    }
  }
}

PlotReprise <- function(data, type, title){
  sample <- subset(data, Type %in% type)%>%
    mutate(Level = factor(Level, levels = paste(seq(5,1,-1))),
           Rate = paste0(as.double(Rate) * 100, "%"),
           Rate = factor(Rate, levels = paste0(seq(0,25,5),"%"))
    )
  
  plot<- ggplot(sample , aes(x=Level, y=Accuracy, fill=Rate))+
    ggtitle(title)+
    geom_bar(stat = "identity", position = position_dodge())+
    geom_text(aes(label=round(Accuracy,2)), position=position_dodge(width=0.9), vjust=-0.25)+
    scale_fill_manual(values=c(Palette_col))+
    guides(fill=guide_legend(nrow=1, byrow=TRUE))+
    theme_custom()
  return(plot)
}
PlotReprise(df_reprise, "Test", "Accuracy en fonction du taux de reprise et du niveau d'aggrégation \n pour la base Test")
PlotReprise(df_reprise, "GU", "Accuracy en fonction du taux de reprise \n pour la base GU")

# TOP K accuracy 

# Stats desc la ou y a le plus d'erreur

# analyse les metrics de la confusion matrix


##### Stats desc ##### 
sample <- df %>%
  group_by(Type, ground_truth_1)%>%
  summarise(
    N = n()
  )%>%
  mutate(Share = N/ifelse(Type =="GU", nrow(df_gu), nrow(df_test)))


ggplot(data = sample, aes(x=ground_truth_1, y=Share, fill=Type))+
  ggtitle('Proportion des différentes catégories')+
  geom_bar(stat = "identity", position = position_dodge())+
  geom_text(aes(label=round(Share*100,1)), position=position_dodge(width=0.9), vjust=-0.25)+
  scale_y_continuous(labels = latex_percent)+
  scale_fill_manual(values=c(Palette_col))+
  theme_custom()


##### Test matrice de confusion niveau 2 GU ##### 
sample <- df_gu

sample <- sample%>%
  mutate(ground_truth_2 =factor(ground_truth_2,levels = sort(unique(sample$ground_truth_2))),
         predictions_2 =factor(predictions_2, levels = sort(unique(sample$ground_truth_2))))

cm <- confusionMatrix(sample$ground_truth_2, df_gu$predictions_2)

cm$table %>%
  data.frame() %>% 
  mutate(Prediction = factor(Prediction, levels =sort(unique(sample$ground_truth_2)))) %>%
  group_by(Reference) %>% 
  mutate(
    total = sum(Freq),
    perc = Freq / total*100
  )%>%  
  ggplot(aes(Prediction, Reference)) +
  geom_tile(aes(fill = perc))+
  scale_fill_gradient(low = "white", high = Palette_col[1]) +
  # geom_text(aes(label = paste0(round(perc,0), "%")), size = 3)+
  theme_custom()+
  guides(fill = "none")  

##### Graphique de reprise des données ##### 

acc_w_rep <- function(sample, q, level){
  N <- nrow(sample)
  model_sample <- sample %>% 
    subset(probabilities >= quantile(probabilities, q))
  n <- nrow(model_sample)
  accuracy <- (sum(model_sample[paste0("ground_truth_",level)] == 
                     model_sample[paste0("predictions_", level)], na.rm = T) + (N-n))/N
  return(accuracy)
}

acc_w_rep(df_gu, 0.1, "2")


df %>% 
  group_by(Type)%>%
  summarise(
    Threshold = quantile(probabilities, seq(0.05, 0.25, 0.05))
  )%>%
  mutate(Q = seq(0.05, 0.25, 0.05))


##### 

data <- df
Type <- c("GU")

TotalSize <- data%>% subset(Type=="GU")

%>%nrow()

Shares <- data%>%
  subset(Type == Type)%>%
  group_by(ground_truth_1, Type)%>%
  summarise(
    N = n(),
  )%>%
  mutate(Share = N/TotalSize*100)

sample <- data%>%
  subset(Type == Type)%>%
  mutate(ground_truth_1 = factor(ground_truth_1, levels = Shares$ground_truth_1),
         predictions_1 = factor(predictions_1, levels = Shares$ground_truth_1))%>%
  select(ground_truth_1, predictions_1)

cm <- confusionMatrix(sample$ground_truth_1, sample$predictions_1)

sample <- cm$table %>%
  data.frame() %>% 
  mutate(Prediction = factor(Prediction, levels = Shares$ground_truth_1)) %>%
  group_by(Reference) %>% 
  mutate(
    total = sum(Freq),
    perc = Freq / total*100
  )


NewTitle <- paste0(sample$Reference, "\n(", rep(round(Shares$Share, 1), each=length(Shares$ground_truth_1)), "%)")
names(NewTitle) <- sample$Reference 

sample <- sample %>%
  mutate(Reference = recode(Reference, !!!NewTitle)
  )

plot <-  ggplot(sample, aes(x=Reference, y= Prediction)) +
  geom_tile(aes(fill = perc))+
  scale_fill_gradient(low = "white", high = Palette_col[1]) +
  geom_text(aes(label = paste0(round(perc,0), "%")), size = 3)+
  theme_custom()+
  guides(fill = "none")  


##### 
topk_accuracy <- function(k, ground_truth_1, predictions_1, predictions_2, predictions_3, predictions_4, predictions_5){
  if (k==1) {
    acc = (ground_truth_1 == predictions_1)
    
  }else if(k==2){
    acc = (ground_truth_1 == predictions_1) | 
      (ground_truth_1 == predictions_2)
    
  }else if(k==3){
    acc = (ground_truth_1 == predictions_1) | 
      (ground_truth_1 == predictions_2) | 
      (ground_truth_1 == predictions_3)
    
  }else if(k==4){
    acc = (ground_truth_1 == predictions_1) | 
      (ground_truth_1 == predictions_2) | 
      (ground_truth_1 == predictions_3) | 
      (ground_truth_1 == predictions_4)
  }else{
    acc = (ground_truth_1 == predictions_1) | 
      (ground_truth_1 == predictions_2) | 
      (ground_truth_1 == predictions_3) | 
      (ground_truth_1 == predictions_4) | 
      (ground_truth_1 == predictions_5)
  }
  return(acc)
  
}
sample <- df %>%
  mutate(Results0 = predictions_5 == ground_truth_5,
         Results1 = topk_accuracy(1, ground_truth_1, predictions_1, predictions_2, predictions_3, predictions_4, predictions_5),
         Results2 = topk_accuracy(2, ground_truth_1, predictions_1, predictions_2, predictions_3, predictions_4, predictions_5),
         Results3 = topk_accuracy(3, ground_truth_1, predictions_1, predictions_2, predictions_3, predictions_4, predictions_5),
         Results4 = topk_accuracy(4, ground_truth_1, predictions_1, predictions_2, predictions_3, predictions_4, predictions_5),
  ) %>%
  group_by(ground_truth_1, Type)%>%
  summarise(
    Accuracy0 = mean(Results0),
    Accuracy1 = mean(Results1),
    Accuracy2 = mean(Results2),
    Accuracy3 = mean(Results3),
    Accuracy4 = mean(Results4),
    
  )

sample
