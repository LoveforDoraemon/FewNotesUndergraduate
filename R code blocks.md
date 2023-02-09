# R code blocks

[TOC]

## Header

```yaml
---
title: "title"
author: "Zijian Cheng"
date: "`r Sys.Date()`"
documentclass: ctexart
geometry: left=2.5cm,right=2cm,top=3cm,bottom=2.5cm
output:
  rticles::ctex:
    fig_caption: yes
    number_sections: yes
    toc: yes
---
```

## init

```R
{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE,comment = "")
```

## ggplot

```R
ggplot(df, aes(x=,y=,fill=,color=)) +
    theme_bw() +
    labs(title="", subtitle="",x=,y=) +
    theme(plot.title=element_text(hjust=0.5), legend.position='none', pane, axis.text.x = element_text(family = "Times", face = "italic",colour = "darkred", size = rel(1.6))) +

    scale_fill_gradient(low=, high=) +
    scale_y_continuous(labels = scales::percent, breaks = seq(0,10,5)) + # 纵坐标百分比

    geom_histogram(bins=, aes(y=..density..,fill=)) +
    geom_density(lwd=0.8, alpha=0.6, fill=) +
    geom_boxplot(col=,width=0.3) +
    stat_boxplot(geom="errorbar",width=0.15,col=,) +
    geom_jitter(width =0.2,size=1.5,color='red') +
    geom_point(size=) +

    geom_vline(xintercept=, color="", linetype="") +
    geom_hline(yintercept=, color="", linetype="") +
```

## cowplot

```R
library(cowplot)
plot_grid(...,nrow=,ncol=)
```

## color

```R
library(RColorBrewer)
"#69b3a2" dark blue

spectral <- brewer.pal(11,'Spectral') # compare

set3 <- brewer.pal(9,'Set3') # discrete
set1 <- brewer.pal(9,'Set1')

blues <- brewer.pal(9,'Blues')
```

## read

```r
df <- read.table("data.txt",header=T)
```

## stat

```R
aov(df$data1~df$data2)
kruskal.test(df$Rate~df$Treatment)

pairwise.t.test(df$data1,df$data2,p.adjust.method="bonferroni")
pairwise.wilcox.test(df$Rate,df$Treatment,p.adjust.method="BH")
```

