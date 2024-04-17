library(ggplot2)
library(reshape2)
library(dplyr)
library(RColorBrewer)

# Read the CSV output
df <- read.csv("output.csv", header=T)

# Produce a scatter plot with an O(n*log(n)) fit over top of it
p <- ggplot(df, aes(x=n, y=firstHit)) + 
       geom_point(size=3,shape=21,fill="firebrick") +
       geom_smooth(method=lm, formula= y~x*log(x), size=1.5, color="firebrick") +
       theme_bw() + ylab("First Hitting Time")

# Display the plot, then save it to a PDF
print(p)
ggsave("output.pdf", width=10, height=7)

# Perform/Display the statistical testing for the O(n*log(n)) fit
# Note the p-value
model = lm(firstHit ~ n*log(n), df)
print(summary(model))