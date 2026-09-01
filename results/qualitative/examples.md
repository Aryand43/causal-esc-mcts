# Qualitative examples (causal_mcts, seed=42)

## Example 1 (esconv-test-0)

**Context:**
- Seeker: hi i am okay, a little bit sad though
- Supporter: Okay. I am very sorry to hear that! Do you want to tell me more about that?
- Seeker: Well with the holidays coming up i have been very stressed and nervous about what i am going to do

**Strategy:** predicted=`Self-disclosure` gold=`Affirmation` (mismatch)

**Generated reply:** It's great to hear that you're feeling down, but it sounds like you've put a lot of effort into planning for the holidays. It can be challenging, especially when everything seems uncertain. Just remember that you're not alone in this

**Gold reply:** I really feel you there. Holidays are so hard.. especially with the way this year has been. Anything specific?

**ROUGE-L:** 0.159 | **BERTScore F1:** 0.731

## Example 2 (esconv-test-1)

**Context:**
- Seeker: Hai. My name is Tom and i really need your help.
- Supporter: ok tell me
- Seeker: I am really sad and stressed up knowing that I am soon losing my job due to the effects of corona virus. For sure if I lose my job now I might go to depression because I have a family to take care of.

**Strategy:** predicted=`Self-disclosure` gold=`Affirmation` (mismatch)

**Generated reply:** I understand how you're feeling, Tom. Losing a job during such an uncertain time can be incredibly tough. It's important to remember that it's okay to seek support from friends, family, or professionals who can offer guidance and practical advice

**Gold reply:** ok, don't worry friend,

**ROUGE-L:** 0.042 | **BERTScore F1:** 0.659

## Example 3 (esconv-test-2)

**Context:**
- Seeker: oh hello
- Supporter: I'm having problems with friends - or more precisely the lack of them. I don't have any.

**Strategy:** predicted=`Self-disclosure` gold=`Restatement` (mismatch)

**Generated reply:** It sounds like you're going through a tough time, and it's important to know that you're not alone. Sometimes, just being open about your feelings can help others understand what you're experiencing. Remember, you're not defined by how

**Gold reply:** Oh I see. You feel as though you don't have any friend's currently? I totally get that. I feel like that a lot too, especially in these weird times we are going through. It's so hard to see people and keep relationships right now

**ROUGE-L:** 0.133 | **BERTScore F1:** 0.764

## Example 4 (esconv-test-3)

**Context:**
- Seeker: Hello
- Supporter: Good Afternoon

**Strategy:** predicted=`Reflection` gold=`Question` (mismatch)

**Generated reply:** Hello! It's nice to meet you too. I'm here to listen and offer any help or guidance you might need. How can I assist you today?

**Gold reply:** How are you doing troday?

**ROUGE-L:** 0.121 | **BERTScore F1:** 0.705

## Example 5 (esconv-test-4)

**Context:**
- Seeker: Hello I am taking a break from my studies to talk. I am fine thanks
- Supporter: Hope you are doing well also
- Seeker: I am just kind of anxious and stressed out with impending academic deadlines

**Strategy:** predicted=`Self-disclosure` gold=`Self-disclosure` (match)

**Generated reply:** It's great that you're taking a break from your studies! That's a common coping mechanism for managing stress during exams or other high-pressure times. Just remember, it's okay to take breaks when you need them. If you ever feel

**Gold reply:** i am fine thanks for asking

**ROUGE-L:** 0.040 | **BERTScore F1:** 0.649

## Example 6 (esconv-test-5)

**Context:**
- Seeker: My friends are trying to shame me.
- Supporter: How ridiculous....

**Strategy:** predicted=`Self-disclosure` gold=`Question` (mismatch)

**Generated reply:** It's understandable that you feel this way, but remember, it's important to prioritize your well-being and self-care. Sometimes, just talking about what's happening can help you feel less alone and more understood. Let's talk about how we

**Gold reply:** hi i am sorry to hear that

**ROUGE-L:** 0.040 | **BERTScore F1:** 0.626

## Example 7 (esconv-test-6)

**Context:**
- Seeker: hello
- Supporter: how are you today?

**Strategy:** predicted=`Self-disclosure` gold=`Question` (mismatch)

**Generated reply:** Hello! I'm just here to listen and offer any help or advice you need. How can I assist you today?

**Gold reply:** Hello, how are you feeling today?

**ROUGE-L:** 0.296 | **BERTScore F1:** 0.760

## Example 8 (esconv-test-7)

**Context:**
- Seeker: I'm feeling pretty good right now.
- Supporter: Is there anything you'd like to talk to me about?
- Seeker: I have a friend who knew me since I was little. She is a little older than me. She still treat me like I'm still little. I tried to explain to her that I'm a grown woman and I grew out of lot of things when I was a kid.

**Strategy:** predicted=`Self-disclosure` gold=`Self-disclosure` (match)

**Generated reply:** It's great that you're sharing this story with us. It sounds like your friend might be experiencing some feelings similar to yours, especially given how much time has passed since you were younger. Sometimes, people can feel like they've aged more

**Gold reply:** Is she listening to you? I know how it feels to be patronised by people who are older.

**ROUGE-L:** 0.161 | **BERTScore F1:** 0.740

## Example 9 (esconv-test-8)

**Context:**
- Seeker: I'm good
- Supporter: How about you?

**Strategy:** predicted=`Self-disclosure` gold=`Question` (mismatch)

**Generated reply:** I'm glad to hear that you're doing well! It's great when people feel supported and understood. If there's anything specific you'd like to talk about or if you need any help with something, please don't hesitate to reach out

**Gold reply:** Very little to complain about today. What brings you here?

**ROUGE-L:** 0.109 | **BERTScore F1:** 0.683

## Example 10 (esconv-test-9)

**Context:**
- Seeker: Hello?
- Supporter: hello. how have you been feeling lately?
- Seeker: Tired
- Supporter: I want to sleep all my problems away

**Strategy:** predicted=`Self-disclosure` gold=`Question` (mismatch)

**Generated reply:** It sounds like you're going through some tough times, and it's great that you're seeking help. Remember, it's okay to feel overwhelmed sometimes. Just take deep breaths, try to relax, and prioritize your needs. Sometimes,

**Gold reply:** how long has this been going on?

**ROUGE-L:** 0.043 | **BERTScore F1:** 0.583

## Example 11 (esconv-test-10)

**Context:**
- Seeker: im very upset
- Supporter: I'm sorry. Would you like to tell me about it?
- Seeker: yes. i invited my friend over to watch my new puppy while i painted my room because my dog has seperation anxiety and can't be alone.
- Supporter: then she showed up with her new boyfriend

**Strategy:** predicted=`Self-disclosure` gold=`Question` (mismatch)

**Generated reply:** It sounds like your friend's arrival was unexpected and made you feel anxious. It's important to remember that everyone is unique and sometimes things happen in ways we don't anticipate. Your friend's presence could have been a bit overwhelming for you,

**Gold reply:** What bothered you about that?

**ROUGE-L:** 0.080 | **BERTScore F1:** 0.674

## Example 12 (esconv-test-11)

**Context:**
- Seeker: Hey.
- Supporter: I am really scared about my job prospects.
- Seeker: I am really fearful about my job situation.

**Strategy:** predicted=`Self-disclosure` gold=`Question` (mismatch)

**Generated reply:** I understand how you're feeling, Seeker. It's important to know that you're not alone in this journey. Many people face similar fears and uncertainties when starting their careers. Remember, every step forward is progress, no matter how small

**Gold reply:** When did this fear start?

**ROUGE-L:** 0.130 | **BERTScore F1:** 0.663

## Example 13 (esconv-test-12)

**Context:**
- Seeker: I am fine thank you. How are you?
- Supporter: I am very well. Do you have any problems you'd like to discuss with me?
- Seeker: Yes. I am having trouble figuring out how to handel a situation with a friend.

**Strategy:** predicted=`Self-disclosure` gold=`Affirmation` (mismatch)

**Generated reply:** It sounds like you're facing some challenges in your friendship. Handing over difficult situations can be challenging, but it's important to approach it thoughtfully and respectfully. Perhaps we could start by discussing what specifically is causing difficulty? What has been

**Gold reply:** Ah, can you give me a few more details? I am sure I will be able to help/

**ROUGE-L:** 0.100 | **BERTScore F1:** 0.677

## Example 14 (esconv-test-13)

**Context:**
- Seeker: Hi, how are you?
- Supporter: Doing good.. How are you?
- Seeker: I'm feeling really anxious these days. I'm finding the COVID online learning experience to be too much for me at this time. I want to stop school, but I don't think I can afford to. I need to get done with school.

**Strategy:** predicted=`Self-disclosure` gold=`Restatement` (mismatch)

**Generated reply:** It sounds like you're going through a tough time, seeking advice from someone who cares about your well-being. It's important to remember that it's okay to feel overwhelmed and there are steps you can take to manage anxiety while still pursuing education

**Gold reply:** I understand your frustration. All of us are challenged due to COVID.

**ROUGE-L:** 0.103 | **BERTScore F1:** 0.693

## Example 15 (esconv-test-14)

**Context:**
- Seeker: Hello. Is anyone there
- Supporter: Hello. How are you today?
- Seeker: I'm doing good. I need some advice

**Strategy:** predicted=`Self-disclosure` gold=`Question` (mismatch)

**Generated reply:** Hello Seeker! It's nice to hear that you're feeling okay today. Sometimes just having someone listen and offer words of encouragement can make all the difference in your mood. If you ever feel overwhelmed or stuck, remember that it's perfectly

**Gold reply:** What can I help you with?

**ROUGE-L:** 0.082 | **BERTScore F1:** 0.635

## Example 16 (esconv-test-15)

**Context:**
- Seeker: Hi
- Supporter: Hello! How are you today?
- Seeker: Ok I guess
- Supporter: I do not know how to tell my husband that I am lonely and I want out of the marriage

**Strategy:** predicted=`Self-disclosure` gold=`Reflection` (mismatch)

**Generated reply:** It sounds like you're going through a tough time, seeking comfort from someone who might be able to offer understanding and support. It's important to remember that your feelings are valid, and it's okay to reach out for help. Let me

**Gold reply:** Oh, that sure is a heavy subject and a heavy thing to be on your mind.

**ROUGE-L:** 0.133 | **BERTScore F1:** 0.723

## Example 17 (esconv-test-16)

**Context:**
- Seeker: I was reallly worried in this suitation
- Supporter: Hi, i'm sean, hows your day going
- Seeker: Was not so good

**Strategy:** predicted=`Self-disclosure` gold=`Question` (mismatch)

**Generated reply:** I understand you're feeling anxious about your situation. It's important to take care of yourself during tough times. Remember, it's okay to feel overwhelmed sometimes, and there are steps you can take to manage stress and anxiety effectively. What specifically

**Gold reply:** Tell me about the situation.

**ROUGE-L:** 0.083 | **BERTScore F1:** 0.698

## Example 18 (esconv-test-17)

**Context:**
- Seeker: evening
- Supporter: How are you feeling at the moment?
- Seeker: i lost those books, I cannot even explain, how it happen

**Strategy:** predicted=`Self-disclosure` gold=`Restatement` (mismatch)

**Generated reply:** I'm sorry to hear about your loss of those books. It must have been a difficult time for you. Remember that it's okay to feel sad or overwhelmed sometimes, and there are people who care about you. Maybe we can find ways

**Gold reply:** So you have lost some books? Were they important to you?

**ROUGE-L:** 0.109 | **BERTScore F1:** 0.723
