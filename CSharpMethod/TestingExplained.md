<style>
  bad {color: red;}
  good {color: green;}
</style>


# Testing Explained

I want list show some cases and what their outputs


## Test Cases

| **Input**              |   | **Ex1**            | **Ex2**                        | **Ex3**                         |
|------------------------|---|--------------------|--------------------------------|---------------------------------|
| **Eiffel Tower**       |   | Machu Picchu       | Statue of Liberty              | <bad>Communism</bad>            |
| **republic**           |   | <bad>Machu Picchu</bad> | Communism                      | <bad>Cicadas</bad>             |
| **<good>ants</good>**               |   | honeybee           | sea turtle                     | tiger                           |
| **<good>Corolla</good>**             |   | Silverado          | Tucson                         | Tesla                           |
| **Public Health Advisors** |   | <bad>whale</bad>   | Environmental Protection Specialists | Intelligence Analyst            |


## Reflection

I had made sure that each of my inputs had at least 5 examples which would be similar in context.

Right now, the similarity search is comparing words rather than semantic meaning. I should make sure that embedding and similarity search is improved.