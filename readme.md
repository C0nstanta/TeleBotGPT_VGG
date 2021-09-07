# GPT3 Model:
<br>
<b>- To get started with the bot, write /start</b>

<b>- To get help, write /help</b>


<table>
<tbody>
<tr>
<td ><img src="/images/img1.png" alt="greeting" width="250" height="450" /></td>
<td ><img src="/images/img2.png" alt="/help" width="250" height="450" /></td>
</tr>
<tr>

</tbody>
</table>

---
## Start GPT3 Menu
In this menu, you can select items such as:
- Start a conversation with the bot
- Clearing dialogs that you previously conducted with the bot
- Go to the menu for changing the basic parameters of GPT3
- Return to main menu

<table>
<tbody>
<tr>
<td align="center"><img src="/images/img3.png" alt="greeting" width="320" height="140" /></td>
 
</tr>
<tr>

</tbody>
</table>

## Options:
**max_length** - Maximum number of tokens, which the model remembers.
The recommended parameter is 256. The maximum is 512. 
If there is more, it will be 512. (not done yet)

**no_repeat_ngram_size** - Penalizes for the number of repetitions.

**do_sample** - The value can be 0 or 1.

If we enable - set 1, then we automatically disable the **top_p** and **top_k** parameters.
If we have 1, we start choosing from the generated words, the next word based on conditional probability.
Increases the likelihood of words that suit us better.

**top_k** - The number of words from which then the one that suits us will be selected.

**top_p** - Same as top_k - only in% (from 0 to 1)
The more% - the fewer words will participate in the further selection to find the desired word.

**temperature** - A value from 0 to 3 (If more, it will be 3).
As Mikhail Konstantinov says - "Temperature is how your model hallucinates."

0 - The model is completely adequate and boring.
3 - delirium - delirium.

**num_return_sequences** - Number of generated return sentences, among which the modeler will then select suitable offer.

**device** - 0 - the model is running on the CPU. 1 - the model is GPU-powered.
We only have a CPU. There are not many options (

**is_always_use_length** - The value can be 0 or 1. If we choose 0, it is generated
answer of any length.

**length_generate** - The length of the sentences generated by the bot.
1 - Short answer.
2 - Average answer.
3 - Long answer.
================================

#Model Vgg19
To work with the model, you first need to upload two images.
The one that will be responsible for the formation of the style.
And the one that will be responsible for the content.

**Upload Style Image** - upload our style.
**Upload Content Image** - uploading our content.

##Options:
The parameters are divided into 2 parts.


###Part 1 - global parameters.

Those - which are used directly when generating a new image.

**epoch number** - the number of epochs.
It usually takes about 500-1000 epochs to generate a normal result.
But on the CPU, we will wait forever for this result.
Therefore, the recommended parameter is 20-50 epochs.
Nothing normal will come of it.
What we can to do without CUDA -?

**show cost** - times in how many epochs we will watch our losses.
But again, this does not work on the bot.
Because until we completely go through the process, the result will not return.

**device** - as it was written above - we only have a CPU.

**image size** - the size of the generated image. The maximum value is 512.


###Part 2 - layer parameters.

For all layers, there are generally accepted parameters that are used to transform the style.
We can easily change these parameters for each layer of the neural network and look at the result.
That is, at will, we can turn on / off each layer of the neural network and see what changes in the generated image.

You can play with all of them and adjust them.
But, how to look at the result if you can't generate a normal image(due to CPU) ... question ...