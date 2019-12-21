
setwd('/Users/gregwalsh/Github/DataScience /DSSA-5101-DATA-EXPLORATION/Project5')

library(tidyverse)
library(ggplot2)
require(scales) 
library(dplyr)

acprint_merge <- function(source_csv, demographic_csv) {
  # Source 
  source_df <- read_csv(
    source_csv,
    col_types = cols(
      Date = col_date("%m/%d/%Y") #convert col to date
    )
  )
  
  # Demographic data sent back
  demographic_df <- read_csv(
    demographic_csv,
    col_types = cols(
      PRINT_DATE = col_date("%m/%d/%Y") #convert col to date
    )
  ) %>% distinct(STUDENT_UNAME, PRINT_DATE, .keep_all = TRUE) #select distinct names and dates
  
  
  merge_df <- merge(source_df, demographic_df, by.x = c("User","Date"), by.y = c("STUDENT_UNAME","PRINT_DATE"), all.x = TRUE) %>%
    # Sanitize data, remove Username
    select(c("Document","Printer","Date","Time","Computer","Pages","Cost", "STVTERM_CODE", "AGE", "SPBPERS_SEX", "MAJR_CODE_LIST", "MINR_CODE_LIST", "CLASS_CODE", "RESIDENT")) %>% 
    arrange(Date)
  
  # clean Resident data
  merge_df <- replace_na(merge_df, list(RESIDENT="N"))
  
  # clean Major list
  merge_df <- merge_df %>% 
    # mark records that will be modified
    mutate(multiple_majors = ifelse(str_detect(MAJR_CODE_LIST, ":"), "Y", "N")) %>%
    # separate records that have multiple majors
    separate_rows(MAJR_CODE_LIST, sep = ":")
  
  # clean Minor list
  merge_df <- merge_df %>% 
    # mark records that will be modified
    mutate(multiple_minors = ifelse(str_detect(MINR_CODE_LIST, ":"), "Y", "N")) %>%
    # separate records that have multiple majors
    separate_rows(MINR_CODE_LIST, sep = ":")
  
  return(merge_df)
}

# Merge acprint4
acprint4_merge <- acprint_merge("acprint4.csv", "EVAN_DSSA_ACPRINT4_3.csv")
# Merge acprint6
acprint6_merge <- acprint_merge("acprint6.csv", "EVAN_DSSA_ACPRINT6_3.csv")

# get unknown majors
enrollment_majors <- c("AFST","ARTS","ARTV","COMM","HIST","LCST","LITT","MAAS","PHIL","BSNS","CMPT","CSCI","CSIS","HTMS","INSY","MBA","CERT","EDOL","MAED","MAIT","TEDU","CERT","LIBA","MAHG","CERT","DNP","DPT","EXSC","HLSC","MSCD","MSN","MSOT","NRS4","NURS","PUBH","SPAD","BCMB","BIOL","CHEM","CPLS","DSSA","ENVL","GEOL","MARS","MATH","MSCP","PHYS","PSM","SSTB","CERT","COUN","CRIM","ECON","MACJ","MSW","POLS","PSYC","SOCY","SOWK","NMAT","UNDC")
acprint6_merge %>% filter(!MAJR_CODE_LIST %in% enrollment_majors) %>% select(MAJR_CODE_LIST) %>% distinct()

#acprint4_merge <- filter(acprint4_merge, undesirable == "deposit")


## Average print 
#hist(acprint4_merge$Pages, xlim = c(0,600), breaks = 50)
#summary(acprint6_merge$Pages)
#barplot(table(acprint4_merge$Pages))
#ggplot(acprint4_merge, aes(x=acprint4_merge$Pages)) + geom_histogram(binwidth=1) + xlim(0, 20) 
#scale_x_continuous(labels = comma)
