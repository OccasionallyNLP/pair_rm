# pairwise rm

# data format
label : chosen sentence index
degree : chosen sentence, rejected sentence difference
```
{'sentence_1': '리처드 닉슨은 인도차이나 전쟁을 끝맺고자 어느나라에서 미군을 철수하였는가? 미국이다. 리처드 닉슨은 미국 대통령으로서 미국의 전쟁 참여를 중단하고 미군을 철수하는 데 성공하였다.',
 'sentence_2': '리처드 닉슨은 인도차이나 전쟁을 끝맺고자 어느나라에서 미군을 철수하였는가? It is difficult to say whether or not the decision to go to war in the Korean War was made by the United States alone or was part of a larger strategy?',
 'label': 0,
 'degree': 2}
```
# 개발 일지
|no|일자|내용|설명|
|---|---|---|---|
|1|23.08.12|margin loss 개발|sample data 검증|
|2|23.08.12|t5forconditionalgeneration rm 개발|sample data 검증|
|3|23.08.12|test_rm 개발 완료|sample data 검증|