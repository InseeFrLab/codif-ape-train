rm(list = ls())
setwd("~/codification-ape/graphs")
library(aws.s3)
library(dplyr)
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

##### Aggrégation par moyenne ##### 
sample <- df %>%
  mutate(Results = predictions_5 == ground_truth_5) %>%
  group_by(ground_truth_1, Type)%>%
  summarise(
    Accuracy = mean(Results),
  )

ggplot(data = sample, aes(x=ground_truth_1, y=Accuracy, fill=Type))+
  ggtitle('Accuracy au niveau 5 (aggrégation par moyenne)')+
  geom_bar(stat = "identity", position = position_dodge())+
  geom_text(aes(label=round(Accuracy,2)), position=position_dodge(width=0.9), vjust=-0.25)+
  scale_fill_manual(values=c(Palette_col))+
  theme_custom()

##### Accuracy niveau 1 #####
sample <- df %>%
  mutate(Results = predictions_1 == ground_truth_1) %>%
  group_by(ground_truth_1, Type)%>%
  summarise(
    Accuracy = mean(Results)
  )

ggplot(data = sample, aes(x=ground_truth_1, y=Accuracy, fill=Type))+
  ggtitle('Accuracy au niveau 1')+
  geom_bar(stat = "identity", position = position_dodge())+
  geom_text(aes(label=round(Accuracy,2)), position=position_dodge(width=0.9), vjust=-0.25)+
  scale_fill_manual(values=c(Palette_col))+
  theme_custom()

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


##### Aggrégation par moyenne avec proportion GU ##### 
sample <- df %>%
  subset(Type =="GU")%>%
  mutate(Results = predictions_5 == ground_truth_5) %>%
  group_by(ground_truth_1, Type)%>%
  summarise(
    Accuracy = mean(Results),
    N = n(),
    )%>%
  mutate(Share = N/ifelse(Type =="GU", nrow(df_gu), nrow(df_test))*100)

NewTitle <- paste0(sample$ground_truth_1, "\n(", round(sample$Share, 1), "%)")
names(NewTitle) <- sample$ground_truth_1 

sample <- sample %>%
  mutate(ground_truth_1 = recode(ground_truth_1, !!!NewTitle)
         )

ggplot(data = sample, aes(x=ground_truth_1, y=Accuracy, fill=Type))+
  ggtitle('Accuracy au niveau 1')+
  geom_bar(stat = "identity", position = position_dodge())+
  geom_text(aes(label=round(Accuracy,2)), position=position_dodge(width=0.9), vjust=-0.25)+
  scale_fill_manual(values=c(Palette_col))+
  theme_custom()

##### Aggrégation par moyenne avec proportion Test ##### 
sample <- df %>%
  subset(Type =="Test")%>%
  mutate(Results = predictions_5 == ground_truth_5) %>%
  group_by(ground_truth_1, Type)%>%
  summarise(
    Accuracy = mean(Results),
    N = n(),
  )%>%
  mutate(Share = N/ifelse(Type =="GU", nrow(df_gu), nrow(df_test))*100)

NewTitle <- paste0(sample$ground_truth_1, "\n(", round(sample$Share, 1), "%)")
names(NewTitle) <- sample$ground_truth_1 

sample <- sample %>%
  mutate(ground_truth_1 = recode(ground_truth_1, !!!NewTitle)
  )

ggplot(data = sample, aes(x=ground_truth_1, y=Accuracy, fill=Type))+
  ggtitle('Accuracy au niveau 1')+
  geom_bar(stat = "identity", position = position_dodge())+
  geom_text(aes(label=round(Accuracy,2)), position=position_dodge(width=0.9), vjust=-0.25)+
  scale_fill_manual(values=c(Palette_col))+
  theme_custom()

##### Accuracy niveau 1 avec proportion GU #####
sample <- df %>%
  subset(Type =="GU")%>%
  mutate(Results = predictions_1 == ground_truth_1) %>%
  group_by(ground_truth_1, Type)%>%
  summarise(
    Accuracy = mean(Results),
    N = n(),
  )%>%
  mutate(Share = N/ifelse(Type =="GU", nrow(df_gu), nrow(df_test))*100)

NewTitle <- paste0(sample$ground_truth_1, "\n(", round(sample$Share, 1), "%)")
names(NewTitle) <- sample$ground_truth_1 

sample <- sample %>%
  mutate(ground_truth_1 = recode(ground_truth_1, !!!NewTitle)
  )

ggplot(data = sample, aes(x=ground_truth_1, y=Accuracy, fill=Type))+
  ggtitle('Accuracy au niveau 1')+
  geom_bar(stat = "identity", position = position_dodge())+
  geom_text(aes(label=round(Accuracy,2)), position=position_dodge(width=0.9), vjust=-0.25)+
  scale_fill_manual(values=c(Palette_col))+
  theme_custom()

##### Accuracy niveau 1 avec proportion Test #####
sample <- df %>%
  subset(Type =="Test")%>%
  mutate(Results = predictions_1 == ground_truth_1) %>%
  group_by(ground_truth_1, Type)%>%
  summarise(
    Accuracy = mean(Results),
    N = n(),
  )%>%
  mutate(Share = N/ifelse(Type =="GU", nrow(df_gu), nrow(df_test))*100)

NewTitle <- paste0(sample$ground_truth_1, "\n(", round(sample$Share, 1), "%)")
names(NewTitle) <- sample$ground_truth_1 

sample <- sample %>%
  mutate(ground_truth_1 = recode(ground_truth_1, !!!NewTitle)
  )

ggplot(data = sample, aes(x=ground_truth_1, y=Accuracy, fill=Type))+
  ggtitle('Accuracy au niveau 1')+
  geom_bar(stat = "identity", position = position_dodge())+
  geom_text(aes(label=round(Accuracy,2)), position=position_dodge(width=0.9), vjust=-0.25)+
  scale_fill_manual(values=c(Palette_col))+
  theme_custom()

##### Matrice de confusion niveau 1 Test #####
df_test <- df_test%>%
  mutate(ground_truth_1 = factor(ground_truth_1, levels = c(LETTERS[1:19], "U")),
         predictions_1 = factor(predictions_1, levels = c(LETTERS[1:19], "U")))

cm <- confusionMatrix(df_test$ground_truth_1, df_test$predictions_1)

cm$table %>%
  data.frame() %>% 
  mutate(Prediction = factor(Prediction, levels = c(LETTERS[1:19], "U"))) %>%
  group_by(Reference) %>% 
  mutate(
    total = sum(Freq),
    perc = Freq / total*100
  )%>%
  ggplot(aes(Prediction, Reference)) +
  geom_tile(aes(fill = perc))+
  scale_fill_gradient(low = "white", high = Palette_col[1]) +
  geom_text(aes(label = paste0(round(perc,0), "%")), size = 3)+
  theme_custom()+
  guides(fill = "none")  

##### Matrice de confusion niveau 1 GU #####
df_gu <- df_gu%>%
  mutate(ground_truth_1 = factor(ground_truth_1, levels = c(LETTERS[1:19], "U")),
         predictions_1 = factor(predictions_1, levels = c(LETTERS[1:19], "U")))

cm <- confusionMatrix(df_gu$ground_truth_1, df_gu$predictions_1)

cm$table %>%
  data.frame() %>% 
  mutate(Prediction = factor(Prediction, levels = c(LETTERS[1:19], "U"))) %>%
  group_by(Reference) %>% 
  mutate(
    total = sum(Freq),
    perc = Freq / total*100
  )%>%
  ggplot(aes(Prediction, Reference)) +
  geom_tile(aes(fill = perc))+
  scale_fill_gradient(low = "white", high = Palette_col[1]) +
  geom_text(aes(label = paste0(round(perc,0), "%")), size = 3)+
  theme_custom()+
  guides(fill = "none")  

##### Test matrice de confusion niveau 2 GU
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
