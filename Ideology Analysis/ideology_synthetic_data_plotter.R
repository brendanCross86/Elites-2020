library(data.table)
library(igraph)
library(Rmisc)
library(dplyr)
library(ggplot2)
library(reshape2)
library(Matrix)
library(ggstance)
library(ggridges)
library(ggpubr)
library(scatterpie)
library(diptest)
library(gtable)
####### GRAPH FOR PAPER ######
color_values=c("#282828","#4F0906","#8F100B","#DB4742","#CFDB00","#4495DB","#0E538F","#082E4F")
binary_colors=c("#7F00DB","#01DB4E")

ideology_file="synthetic_ideology_results.RDS"
interaction_file="synthetic_interaction_matrix.RDS"
#### LOAD IDEOLOGY ESTIMATES ####
res=readRDS(ideology_file)
#### LOAD INTERACTION MATRIX ####
interaction_matrix=readRDS(interaction_file)
###### LOAD INFLUENCER LIST ######
influencers_data=fread("syntethic_influencers_data.csv")

top_5_cat=influencers_data[rank<6,.(id,name,type=type,rank)]

influencers_data$n_type[influencers_data$type=="far-right"]=1
influencers_data$n_type[influencers_data$type=="right"]=0.66
influencers_data$n_type[influencers_data$type=="lean-right"]=0.33
influencers_data$n_type[influencers_data$type=="center"]=0
influencers_data$n_type[influencers_data$type=="lean-left"]=-0.33
influencers_data$n_type[influencers_data$type=="left"]=-0.66
influencers_data$n_type[influencers_data$type=="far-left"]=-1
influencers_data$n_type[influencers_data$type=="fake"]=1.33


influencers_data$type=factor(influencers_data$type,
                             levels = c("fake","far-right","right","lean-right","center",
                                        "lean-left","left","far-left"),
                             labels = c("Fake news", "Extreme bias right", "Right news","Right leaning News",
                                        "Center news", "Left leaning news","Left news", "Extreme bias left"))
top_5_cat_ranks=as.data.frame.matrix(xtabs(1/rank ~ id + type, influencers_data))
top_5_cat_ranks$id=rownames(top_5_cat_ranks)
### ADD CONTINUOUS COLOR FOR NAMES ####
color_numeric_vector=c(1.3,0.99,0.66,0.33,0,-0.33,-0.66,-0.99)
top_5_cat_ranks$color_score=as.numeric(xtabs(1/rank ~ id + type, influencers_data)%*%color_numeric_vector/rowSums(xtabs(1/rank ~ id + type, influencers_data)))


influencers_data=influencers_data[,.(rank=min(rank),
                                     name=unique(name),
                                     type=.SD[rank==min(rank), type][1],
                                     n_type=.SD[rank==min(rank), n_type][1]),
                                  by=.(id)]
####### CHECK THE ACCORDANCY BETWEEN PHI AND OUR CLASSIFICATION ####
res_infl_data=data.table(id=as.character(res$colnames),phi=as.numeric(scale(res$colcoord)))

found_influencers=merge(influencers_data,res_infl_data, by.x="id", by.y = "id")

setDT(top_5_cat_ranks)
top_5_cat_ranks=top_5_cat_ranks[id %in% top_5_cat$id]
top_5_cat_ranks=merge(top_5_cat_ranks,top_5_cat[,.(name=unique(name)),by=.(id)], by="id")
interaction_matrix=Matrix(interaction_matrix, sparse = TRUE)
summ <- summary(interaction_matrix)

df=data.table(user      = rownames(interaction_matrix)[summ$i],
              influencer = colnames(interaction_matrix)[summ$j],
              Weight      = summ$x)


df=merge(df,influencers_data[,.(id,n_type, type)], by.x="influencer", by.y = "id")

user_stat=df[,.(n_of_links=.N,leaning=mean(n_type, na.rm = T)),by=.(user)]
user_phi=data.table(id=res$rownames, phi=as.numeric(scale(res$rowcoord)))
df=merge(df,user_phi, by.x="user",by.y="id")

data_to_plot=data.table(id=res$rownames,phi=as.numeric(scale(res$rowcoord)))
data_to_plot=merge(data_to_plot,user_stat,by.x="id", by.y = "user")
print(cor.test(data_to_plot[!is.na(leaning)&!is.na(phi),leaning],
               data_to_plot[!is.na(leaning)&!is.na(phi),phi],method = "pearson"))

#### IF THEY ARE NOT ACCORDING, CHANGE ORIENTATION OF IDEOLOGY ####

if (cor(data_to_plot[!is.na(leaning)&!is.na(phi),leaning],
        data_to_plot[!is.na(leaning)&!is.na(phi),phi],method = "pearson")<0)
{
  data_to_plot$phi=-1*data_to_plot$phi
  df$phi=-1*df$phi
  found_influencers$phi=-1*found_influencers$phi
  res_infl_data$phi=-1*res_infl_data$phi
  user_phi$phi=-1*user_phi$phi
  
}

#### CREATE STATS AND CALCULATRE VALUES FOR THE PLOT NAMES ####
influencers_avg_audience=df[,.(count=.N,
                               mean_phi=mean(phi),med_phi=median(phi),
                               min_val=min(phi),
                               max_val=max(phi),
                               q_1 = quantile(phi,probs=0.25),
                               q_3 = quantile(phi, probs = 0.75),
                               sd_ph=sd(phi),
                               up=quantile(phi,probs=0.95),
                               low=quantile(phi,probs=0.05)),
                            by=.(id=influencer)]
influencers_avg_audience$iqr=influencers_avg_audience$q_3-influencers_avg_audience$q_1

influencers_avg_audience=merge(influencers_avg_audience,influencers_data, by.x="id", by.y="id")
influencers_avg_audience=merge(influencers_avg_audience,res_infl_data[,.(id,infl_phi=phi)], by.x="id", by.y="id")

top_5_data_to_plot=influencers_avg_audience[id %in% top_5_cat$id]
top_5_data_to_plot=top_5_data_to_plot[order(med_phi, decreasing = T)]
top_5_data_to_plot$y=seq(from = 0, to = 4, by =4/nrow(top_5_data_to_plot))[1:nrow(top_5_data_to_plot)]


df_to_plot=df
df_to_plot$type="all"
df_to_plot=rbind(df_to_plot,df)

### BOXPLOT STATS ####
top_05_influencers_users=df[influencer %in% top_5_cat$id]
top_05_influencers_users=merge(top_05_influencers_users,top_5_data_to_plot[,.(id,y)], by.x="influencer",by.y="id")
### add space for text ###
top_5_data_to_plot$xpos=ifelse(top_5_data_to_plot$med_phi<0,
                               top_5_data_to_plot$low,
                               top_5_data_to_plot$up)

top_5_data_to_plot$xpos=ifelse(top_5_data_to_plot$med_phi<0,
                               top_5_data_to_plot$xpos-0.1,
                               top_5_data_to_plot$xpos+0.1)
data_for_pies=top_5_cat_ranks
data_for_pies=merge(data_for_pies,top_5_data_to_plot[,.(id,y,xpos, med_phi)], by="id")
setDT(data_for_pies)

##### DIST STAT ###
all_distribution=rbind(df[,.(phi=unique(phi),type="users"), by=.(id=user)],
                       influencers_avg_audience[,.(phi=med_phi,type="influencers"), by=.(id)])

####### DIPTESTS ######
dip.test(all_distribution[type=="influencers",phi])
dip.test(all_distribution[type=="users",phi])

dt_for_plot_bx=top_05_influencers_users[,.(count=.N,
                                           min=quantile(phi,0.05),
                                           low=quantile(phi,0.25),
                                           med=quantile(phi,0.5),
                                           up=quantile(phi,0.75),
                                           max=quantile(phi,0.95)),
                                        by=.(y,type,n_type, influencer)]


bp_2016=ggplot()+
  geom_boxploth(data=dt_for_plot_bx,
                aes(y=y,xmin=min,xmax=max,xmiddle=med,
                    xlower=low,xupper=up,
                    color=type,
                    group=y),
                stat="identity", show.legend = F)+
  geom_text(data=data_for_pies[med_phi<0], aes(x=xpos-0.1,y=y,label=name), color="black", show.legend = F, hjust = 1)+
  geom_text(data=data_for_pies[med_phi>0], aes(x=xpos+0.1,y=y,label=name), color="black", show.legend = FALSE, hjust = 0)+
  ## add count for fucking stupid nature ###
  geom_text(data=dt_for_plot_bx[med<0], aes(x=max+0.1,y=y,label=paste0("n=",count)), color="black", show.legend = F, hjust = 0)+
  geom_text(data=dt_for_plot_bx[med>0], aes(x=min-0.1,y=y,label=paste0("n=",count)), color="black", show.legend = FALSE, hjust = 1)+
  ####
  geom_scatterpie(aes(x=xpos, y=y),
                  data=data_for_pies, cols=c("Fake news",
                                             "Extreme bias right","Right news","Right leaning News",
                                             "Center news",
                                             "Left leaning news","Left news","Extreme bias left"), 
                  color=NA, 
                  pie_scale=0.9,
                  alpha=0.9)+
  theme_classic()+
  scale_x_continuous(limits = c(-2.2,2.2))+
  scale_color_manual(values=color_values)+
  scale_fill_manual(values=color_values,labels=c("fake news","extreme bias right","right news","right leaning news","center news",
                                                 "left leaning news","left news","extreme bias left"))+
  theme(legend.position = c(0.5,0.98),
        legend.direction="horizontal",
        legend.background = element_rect(fill = NA, colour = NA),
        legend.title = element_blank(),
        axis.line.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.line.x = element_blank(),
        axis.ticks.x =element_blank(),
        axis.text.x = element_blank(),
        text=element_text(size=14),
        plot.margin=unit(c(0,0,0,0), "cm")
  )+
  labs(x="Latent Ideology Scale")+
  guides(colour = guide_legend(nrow = 2,
                               keywidth = unit(1, "cm"),
                               keyheight = unit(1, "cm"),
                               override.aes = list(shape = 16,
                                                   size = 6)),
         fill=guide_legend(nrow = 2))+
  coord_fixed()

###### DISTRIBUTION #####
infl_dens=density(all_distribution[type=="influencers", phi], n=2048, bw="SJ")
usrs_dens=density(all_distribution[type=="users", phi], n=2048, bw="SJ")

density_est=rbind(data.table(type="influencers", x=infl_dens$x, y=infl_dens$y),
                  data.table(type="users", x=usrs_dens$x, y=usrs_dens$y))

#### BARPLOT #### 
dens_2016=ggplot(all_distribution, aes(x=phi,fill=type))+
  #geom_area(alpha=0.6)+
  geom_histogram(binwidth = 0.05,alpha=0.5, aes(y = ..density..), position = "identity")+
  #scale_y_log10()+#, aes(y = ..density..), position = "identity")+
  xlim(-2.2,2.2)+
  scale_y_continuous(breaks = seq(0, 9, by = 3), limits = c(0,10))+
  theme_classic()+
  scale_fill_manual(values = binary_colors)+
  theme(legend.title = element_blank(),#)+#,
        legend.position = c(0.6,0.7),
        text=element_text(size=14),
        plot.margin=unit(c(0,0,0,0), "cm")
  )+
  labs(x="Latent ideology", y="density")

ggarrange(bp_2016,dens_2016,nrow=2, heights = c(2, 0.9), align = "v")
