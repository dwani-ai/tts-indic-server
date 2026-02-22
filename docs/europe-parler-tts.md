

{
  "input": "Salut toi, comment vas-tu aujourd'hui?",
  "voice": "Daniel's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise.",
  "model": "parler-tts/parler-tts-mini-multilingual-v1.1",
  "response_format": "mp3",
  "speed": 1
}


curl -X 'POST' \
  'https://slabstech-dhwani-internal-api-server.hf.space/v1/audio/speech' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": "Salut toi, comment vas-tu aujourd hui?",
  "voice": "Daniel s voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise.",
  "model": "parler-tts/parler-tts-mini-multilingual-v1.1",
  "response_format": "mp3",
  "speed": 1
}' -o french.mp3



| Language   | Speaker Name | Number of occurrences it was trained on |
|------------|--------------|-----------------------------------------|
| Dutch      | Mark         | 460066                                  |
|            | Jessica      | 4438                                    |
|            | Michelle     | 83                                      |
| French     | Daniel       | 10719                                   |
|            | Michelle     | 19                                      |
|            | Christine    | 20187                                   |
|            | Megan        | 695                                     |
| German     | Nicole       | 53964                                   |
|            | Christopher  | 1671                                    |
|            | Megan        | 41                                      |
|            | Michelle     | 12693                                   |
| Italian    | Julia        | 2616                                    |
|            | Richard      | 9640                                    |
|            | Megan        | 4                                       |
| Polish     | Alex         | 25849                                   |
|            | Natalie      | 9384                                    |
| Portuguese | Sophia       | 34182                                   |
|            | Nicholas     | 4411                                    |
| Spanish    | Steven       | 74099                                   |
|            | Olivia       | 48489                                   |
|            | Megan        | 12                                      |