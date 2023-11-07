# Get started
## Introduction
LangChain是一个用于开发基于语言模型的应用程序的框架。它使应用程序具备以下特点：  
- 具有上下文感知性：可以将语言模型连接到上下文的数据源（提示指令、少量示例、	内容以构建响应等）。  
- 具备推理能力：依赖语言模型进行推理（根据提供的上下文来决定如何回答，采取什	么行动等）。    
该框架由多个部分组成，您可以查看下面的部分以了解它们之间的交互：  

这些部分包括：  
- [LangChain Packages]：Python 和 JavaScript 包。包含各种组件的接口和集成，用于	基本运行时以将这些组件组合成链和代理，并提供现成的链和代理的实现。
- LangChain Templates：一组易于部署的参考架构，适用于各种任务。
- LangServe：用于将 LangChain 链部署为 REST API 的库。
- LangSmith：开发平台，允许您在任何 LLM 框架上构建的链进行调试、测试、评估和	监控，并与 LangChain 无缝集成。
总的来说，这些产品简化了整个应用程序的生命周期：
- **开发Develop**：使用 LangChain/LangChain.js 编写您的应用程序。可以使用参考模板快速启动。
- **生产Productionize**：使用 LangSmith 检查、测试和监视您的链，以便不断改进并自信地部署。
- **部署Deploy**：使用 LangServe 将任何链转换为 API。  

## LangChain Packages  
LangChain packages 的主要价值在于：
- 组件：用于处理语言模型的可组合工具和集成。组件是模块化且易于使用的，无论您是	否使用 LangChain 框架的其他部分。
- 现成链：用于完成更高级任务的内置组件组装。
现成链使入门变得更容易。组件使定制现有链和构建新链变得更容易。

## Get started  
以下是安装 LangChain、设置您的环境并开始构建的方法。
我们建议按照我们的快速入门指南，通过构建您的第一个 LangChain 应用程序来熟悉该框架。  

## Modules
LangChain 为以下模块提供标准且可扩展的接口和集成，从最简单到最复杂的列出如下：

模型输入/输出：与语言模型进行接口。
检索：与特定于应用程序的数据进行接口。

链：构建调用序列。

代理：让链根据高级指令选择要使用的工具。

存储：在链的运行之间保留应用程序状态。

回调：记录和流式传输任何链的中间步骤。

## Examples, ecosystem, and resources​
### Use cases
针对常见的端到端用例，如文档问题回答、聊天机器人、分析结构化数据等，提供演练和技术。
### 指南
使用 LangChain 开发的最佳实践。
### 生态系统 
LangChain 是一个丰富的工具生态系统的一部分，与我们的框架集成并在其之上构建。查看我们不断增长的集成和依赖仓库列表。
### 社区
前往社区导航器，找到提问、分享反馈、与其他开发人员交流和探讨 LLM 未来的地方。
## API 参考
前往参考部分，查看 LangChain Python 包中所有类和方法的完整文档。

Quicstart
Security



# Installation

# 官方发布
要安装 LangChain，请运行以下命令：

使用 Pip：
```sh
pip install langchain
```

使用 Conda：
```sh
conda install -c langchain langchain
```

这将安装 LangChain 的最低要求。LangChain 的许多价值体现在与各种模型提供商、数据存储等集成时。默认情况下，不会安装所需的集成依赖项。您需要单独安装特定集成的依赖项。

从源代码安装
如果您想要从源代码安装，可以通过克隆仓库，并确保目录在 PATH/TO/REPO/langchain 下，然后运行以下命令：

```sh
pip install -e .
```

# 快速入门

## 安装
要安装 LangChain，请运行以下命令：

使用 Pip：
```sh
pip install langchain
```
## Environment setup

LangChain 通常需要与一个或多个模型提供商、数据存储、API 等进行集成。在本示例中，我们将使用 OpenAI 的模型 API。

首先，需要安装他们的 Python 包：
```sh
pip install openai
```

要访问 API，需要 API 密钥，您可以通过创建帐户并前往指定页面获取。获取 API 密钥后，可以将其设置为环境变量：
```sh
export OPENAI_API_KEY="YOUR_API_KEY"
```

如果您不想设置环境变量，您还可以在初始化 OpenAI LLM 类时直接通过 `openai_api_key` 参数传递密钥：
```python
from langchain.llms import OpenAI

llm = OpenAI(openai_api_key="YOUR_API_KEY")
```

## LangSmith Setup
LangChain 应用程序通常包含多个步骤，需要多次调用 LLM。随着这些应用程序变得越来越复杂，能够检查链或代理内部发生了什么变得至关重要。这时，LangSmith 就非常有用。

请注意，LangSmith 不是必需的，但它非常有帮助。如果您想使用 LangSmith，请在上面的链接上注册，并确保设置您的环境变量以开始记录跟踪：
```sh
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="YOUR_API_KEY"
```

## Building an application构建应用程序
现在，我们可以开始构建我们的语言模型应用程序。LangChain 提供了许多模块，可用于构建语言模型应用程序。这些模块可以作为简单应用程序的独立部分使用，也可以组合用于更复杂的用例。

LangChain 帮助创建的最常见和最重要的链包括以下三个组成部分：

1. LLM（Language Model）：语言模型是核心的推理引擎。为了使用 LangChain，您需要了解不同类型的语言模型以及如何与它们一起使用。
2. 提示模板（Prompt Templates）：这些模板为语言模型提供指令。它控制语言模型的输出，因此理解如何构建提示和不同提示策略至关重要。
3. 输出解析器（Output Parsers）：这些解析器将 LLM 的原始响应转换为可用于下游处理的格式。

在本入门指南中，我们将首先单独介绍这三个组件，然后讨论如何将它们组合在一起。了解这些概念将有助于您使用和定制 LangChain 应用程序。大多数 LangChain 应用程序允许您配置 LLM 和/或使用的提示，因此知道如何利用这一点将是一个重要的优势。

## LLM（语言模型）
在 LangChain 中，有两种类型的语言模型，称为：

1. LLM（Language Model）：这是一个以字符串作为输入并返回字符串作为输出的语言模型。
2. ChatModel（聊天模型）：这是一个以消息列表作为输入并返回消息作为输出的语言模型。

LLM 的输入/输出简单易懂，就是一个字符串。但聊天模型的输入是消息列表，输出是单个消息。消息对象包括两个必需组件：

- content：消息的内容。
- role：消息所来自的实体的角色。

LangChain 提供了几种对象，用于轻松区分不同的角色：

- HumanMessage：来自人类/用户的消息。
- AIMessage：来自 AI/助手的消息。
- SystemMessage：来自系统的消息。
- FunctionMessage：来自函数调用的消息。

如果上述角色都不合适，还可以使用 ChatMessage 类手动指定角色。要了解如何最有效地使用这些不同的消息，可以查看我们的提示指南。

LangChain 提供了一个通用接口，适用于 LLM 和 ChatModel。但了解这些区别对于构建特定语言模型的提示很有帮助。

LangChain 提供的标准接口具有两个方法：

1. predict：接受字符串作为输入，返回字符串。
2. predict_messages：接受消息列表作为输入，返回消息。

让我们看看如何使用这些不同类型的模型和不同类型的输入。首先，让我们导入一个 LLM 和一个 ChatModel：

```from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

llm = OpenAI()
chat_model = ChatOpenAI()

llm.predict("hi!")
>>> "Hi"

chat_model.predict("hi!")
>>> "Hi"
```

OpenAI 和 ChatOpenAI 对象实际上只是配置对象。您可以使用参数（如温度等）初始化它们，并传递它们。

接下来，让我们使用 `predict` 方法运行字符串输入：

```python
text = "What would be a good company name for a company that makes colorful socks?"

llm.predict(text)
# >> "Feetful of Fun"

chat_model.predict(text)
# >> "Socks O'Color"
```

最后，让我们使用 `predict_messages` 方法运行消息列表：

```python
from langchain.schema import HumanMessage

text = "What would be a good company name for a company that makes colorful socks?"
messages = [HumanMessage(content=text)]

llm.predict_messages(messages)
# >> "Feetful of Fun"

chat_model.predict_messages(messages)
# >> "Socks O'Color"
```

对于这两种方法，您还可以通过关键字参数传递参数。例如，您可以传递 temperature=0 来调整在运行时使用的温度，从而覆盖对象在配置时设置的任何值。

## Prompt templates提示模板 

大多数 LLM 应用程序不会直接将用户输入传递给 LLM。通常，它们将用户输入添加到更大的文本片段中，称为提示模板（Prompt Template），以提供有关具体任务的附加上下文。
在上面的示例中，我们向模型传递的文本包含生成公司名称的指令。对于我们的应用程序，用户只需提供公司/产品的描述，而无需担心提供模型指令。


PromptTemplates 可以帮助实现这一点！它们将从用户输入到完全格式化的提示的所有逻辑捆绑在一起。这可以从非常简单开始，例如，生成上面字符串的提示可能如下所示：
```
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")

prompt.format(product="colorful socks")
```
```
What is a good name for a company that makes colorful socks?
```
然而，使用它们的好处不止如此。您可以“局部”处理变量，例如，您可以一次只格式化某些变量。您可以将它们组合在一起，轻松地将不同的模板组合成单个提示。有关这些功能的详细信息，请参阅提示部分。


PromptTemplates 也可以用于生成消息列表。在这种情况下，提示不仅包含有关内容的信息，还包括每条消息（其角色、在列表中的位置等）的信息。在这里，最常见的情况是 ChatPromptTemplate 是 ChatMessageTemplates 的列表。每个 ChatMessageTemplate 包含了如何格式化该 ChatMessage 的指令 - 其角色以及内容。让我们看一下下面的示例：
```
from langchain.prompts.chat import ChatPromptTemplate

template = "You are a helpful assistant that translates {input_language} to {output_language}."

human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")
```
ChatPromptTemplates 也可以以其他方式构建 - 有关更多详细信息，请参阅提示部分。
## Output parsers输出解析器 

OutputParsers 将 LLM 的原始输出转换为可用于下游处理的格式。有几种主要类型的 OutputParsers，包括：
- 将 LLM 中的文本转换为结构化信息（例如 JSON）
- 将 ChatMessage 转换为普通字符串
- 将除消息之外的调用返回的额外信息（如 OpenAI 函数调用）转换为字符串
有关完整信息，请参阅输出解析器部分。
在这个入门指南中，我们将编写自己的输出解析器 - 一个将逗号分隔的列表转换为列表的解析器：
```
from langchain.schema import BaseOutputParser
class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

CommaSeparatedListOutputParser().parse("hi, bye")# >> ['hi', 'bye']
```
## PromptTemplate + LLM + OutputParser 

现在，我们可以将所有这些组合成一个链。该链将接受输入变量，将这些变量传递给提示模板以创建提示，将提示传递给语言模型，然后通过（可选的）输出解析器传递输出。这是一种方便的方式，可以将模块化的逻辑打包在一起。让我们看看它是如何运作的！

```
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser

class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""


    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])
chain = chat_prompt | ChatOpenAI() | CommaSeparatedListOutputParser()
chain.invoke({"text": "colors"})
# >> ['red', 'blue', 'green', 'yellow', 'orange']
```

请注意，我们使用 | 语法来将这些组件连接在一起。这 | 语法被称为 LangChain 表达语言。要了解有关此语法的更多信息，请阅读文档。

## Next steps接下来的步骤
这就是全部！我们已经介绍了如何创建 LangChain 应用程序的核心构建块。在所有这些组件（LLMs、提示、输出解析器）中还有更多的细微差别，还有更多不同的组件需要学习。要继续您的学习之旅：

- 深入了解 LLMs、提示和输出解析器
- 学习其他关键组件
- 阅读有关 LangChain 表达语言的信息，以了解如何将这些组件链接在一起
- 查看我们有关特定主题的详细演练的有用指南
- 探索端到端用例



# Security
LangChain拥有与各种外部资源（如本地和远程文件系统、API和数据库）的大型生态系统集成。这些集成使开发人员能够创建多功能应用程序，将LLMs的强大功能与访问、互动和操作外部资源的能力相结合。
## Best Practices
在构建此类应用程序时，开发人员应遵循良好的安全实践：

- 限制权限：将权限明确定义为应用程序的需求。授予广泛或过多的权限可能会引入重大安全漏洞。为了避免这类漏洞，考虑使用只读凭据、禁止访问敏感资源，根据您的应用程序需要，使用沙盒技术（如在容器内运行）等。

- 预期潜在的滥用：正如人类可能犯错误一样，大型语言模型（LLMs）也可能出现问题。始终假设系统访问或凭据可能以其被分配的权限允许的任何方式使用。例如，如果一对数据库凭据允许删除数据，最安全的做法是假设任何能够使用这些凭据的LLM实际上可以删除数据。

- 深度防御：没有绝对完美的安全技术。微调和良好的链设计可以降低LLM出错的概率，但不能完全消除。最好的做法是结合多层安全方法，而不是依赖单一防御层来确保安全。例如，同时使用只读权限和沙盒技术，以确保LLMs只能访问明确为它们使用的数据。

不遵循这些实践可能会导致以下风险，但不限于：

- 数据损坏或丢失。
- 未经授权访问机密信息。
- 临时关键资源性能或可用性受损。
- 示例场景和缓解策略：

Example scenarios with mitigation strategies:示例场景和缓解策略：

- 用户可能会要求具有文件系统访问权限的代理删除不应删除的文件或读取包含敏感信息的文件的内容。为了缓解风险，限制代理仅能够使用特定目录，并仅允许其读取或写入安全的文件。考虑通过在容器中运行代理来进一步隔离它。

- 用户可能会要求具有对外部API的写入权限的代理将恶意数据写入API，或从API中删除数据。为了缓解风险，为代理提供只读API密钥，或限制其仅能够使用已经对此类滥用具有抵抗力的终端。

- 用户可能会要求具有对数据库的访问权限的代理删除表格或改变模式。为了缓解风险，将凭证的范围限定为代理需要访问的表格，考虑颁发只读凭证。

如果您正在构建访问外部资源（如文件系统、API或数据库）的应用程序，请考虑与公司的安全团队沟通，以确定如何最好地设计和保护您的应用程序。

## 漏洞报告
请通过电子邮件发送安全漏洞报告至 security@langchain.dev。这将确保问题得到及时分派和处理。

## 企业解决方案
LangChain可能为具有额外安全要求的客户提供企业解决方案。请通过 sales@langchain.dev 与我们联系。


# LangChain Expression LanguageLangChain表达语言

为了尽可能简化创建自定义链的过程，我们实现了一个"Runnable"协议。大多数组件都实现了"Runnable"协议。这是一个标准接口，使得定义自定义链以及以标准方式调用它们变得容易。标准接口包括以下方法：

- stream：流式返回响应的数据块
- invoke：在输入上调用链
- batch：在输入列表上调用链

这些方法也有相应的异步方法：

- astream：异步流式返回响应的数据块
- ainvoke：异步在输入上调用链
- abatch：异步在输入列表上调用链
- astream_log：异步流式返回中间步骤，以及最终响应

输入类型因组件而异。

各组件的输入类型和输出类型如下：

**输入类型：**

- Prompt（提示）：字典（Dictionary）
- Retriever（检索器）：单个字符串（Single string）
- LLM、ChatModel：单个字符串、聊天消息列表或PromptValue
- Tool（工具）：根据工具的不同，可以是单个字符串或字典

**输出类型：**

- LLM：字符串（String）
- ChatModel：ChatMessage
- Prompt：PromptValue
- Retriever：文档列表
- Tool：根据工具的不同而不同
- OutputParser：根据解析器的不同而不同

所有可运行组件都公开了用于检查输入和输出的输入模式和输出模式：

- input_schema：从Runnable结构自动生成的输入Pydantic模型
- output_schema：从Runnable结构自动生成的输出Pydantic模型

让我们看一下这些方法。为此，我们将创建一个非常简单的PromptTemplate（提示模板）+ ChatModel（聊天模型）链。
```
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
chain = prompt | model
```




