# Approaching Large Textual Archives as Big Data
Full Report: https://docs.google.com/document/d/1fkNEOQqDBmi-F9thsPz423VFHPXYKs0M3lujP_UIALQ/

# Abstract

In recent years, declassified documents from Communist bloc countries have produced substantial textual archives about the Cold War. Big data methods are useful to glean patterns from these archives, which can then be used to support or question existing historical narratives about key events. 

In this paper, we study a corpus of telegrams made available by the Wilson Center Digital Archive, which were sent among Communist allies in the lead up, during and the aftermath of the Korean War. We focus on a time-based analysis, analysing cross-sections of the data by time periods and noting the shifts in sentiment and word associations. 

These shifts shed light on some unresolved questions about the Korean War: how did attitudes and relationships between the three Communist players of China, Korea and the Soviet Union change as the Korean War progressed?

# Methodology
![image](https://user-images.githubusercontent.com/38340979/151695990-26402c86-8508-4cac-8497-898bd1308f96.png)

# Results
## Sentiments of Telegrams between China and the USSR
![image](https://user-images.githubusercontent.com/38340979/151695976-15122ac7-8e24-41d2-b104-21ffa272e969.png)

The results of the sentiment analysis of the telegrams give us significant insight into the relationship between China and the USSR over the course of the war. The rise and fall in positive sentiments in telegrams could be attributed and traced back to events in the war, such as when China sent troops to North Korea with the USSR’s support, positive sentiments were high as reflected in the telegrams from USSR to China; or when the Korean People Army suffered setback and significant losses upon the Battle of Inchon, the positive sentiments dropped for those events contribute to a build up in tension between the two countries as the war progresses.
The percentage of positive sentiments in telegrams from the USSR to China did not increase back to its previous peak back in July 1950. This provided an indication of the deteriorating relationship between China and USSR over the course of the Korean War. Given that the sentiments of telegrams from USSR to China decreased steadily over the course of the war, it can be deduced that the start of the eventual Sino-Soviet split can be traced back to late 1950. It is also during this time that Stalin did not provide China with the air cover that resulted in massive Chinese casualties. As mentioned above, Stalin had written to China, vaguely promising that they “will try to provide the air cover of these units.” (Telegram, Stalin to Mao Zedong, 5 July 1950) However, the Chinese army had gone into combat in late October 1950 without air cover or bomber support, and this “alleged betrayal” by Stalin was a critical point in the eventual breakdown of Sino-Soviet relations (Mark O’Neill, 2000).
Given that the sentiments of telegrams from China to USSR remain considerably positive up till January 1953, it shows that despite Stalin’s lack of support and air cover, Mao chose not to display his displeasure with Stalin, and that the Sino-Soviet split would not occur until after Stalin’s death. Whatever unhappiness Mao had towards the USSR may not have been reflected in the telegrams, but they may have led to suppressed resentment that was released after Stalin’s death. Besides Mao’s distaste for Nikita Khruschev, this resentment was a crucial factor for the eventual Sino-Soviet split. 

## Changes in Word Cloud Representations
### Lead up to Korean War (Jan-May 1950)
![image](https://user-images.githubusercontent.com/38340979/151696139-e13fab5c-9ce5-4837-a31a-f67205267b2d.png)
### North Korean Offensive (May-Sep 1950)
![image](https://user-images.githubusercontent.com/38340979/151696168-48ba7e8c-b90e-441c-9a26-31917c90cb19.png)
### American Advance (Sep-Dec 1950)
![image](https://user-images.githubusercontent.com/38340979/151696176-d18622ca-4167-4948-8fe1-0a5a3a750511.png)

In summary, the word vector representations together with t-SNE dimensionality reduction provides a big picture of the documents used to train the model, akin to a visual summary. This allowed our group to explore the telegrams like a geographical map. There are certainly limitations to what word2vec can provide, in addition there is also a huge reliance on having a sufficient amount of training data. The assumptions and conclusions drawn from the visualisations depend on the interpreter of the data, limited by the knowledge they possess. 
However, these word representations our group provided show a contemporary way of visualising data, revealing the relationships between various key players and events in the Korean War. We observed how the Communist bloc leaders had a close relationship during the lead up to the Korean War, and how this close relationship was strained by a lack of Soviet aid in the American Advance.
