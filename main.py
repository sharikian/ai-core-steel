import telebot
import pandas as pd
import uuid
import random
import os
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, InlineQueryResultArticle, InputTextMessageContent
from g4f import ChatCompletion, Provider
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure providers
ACTIVE_PROVIDERS = [
    Provider.Copilot,
    Provider.Yqcloud,
    Provider.ChatGptEs,
    Provider.PollinationsAI,
    Provider.Glider,
    Provider.Liaobots,
    Provider.Phind,
]

# Global cache for models: [{"name": "gpt-4", "state": "enable|disable|checking", "rate": response_time}]
MODELS_TO_CHECK = []

# Global dictionary to track cancellation requests
cancel_flags = {}

class AutoProvider:
    def __init__(self, providers):
        self.providers = providers
        self.current_provider = None
        self.last_failure = {}
        self.retry_delay = 300  # 5 minutes in seconds

    def get_provider(self):
        current_time = time.time()
        if self.current_provider:
            last_fail_time = self.last_failure.get(self.current_provider.__name__, 0)
            if current_time - last_fail_time > self.retry_delay:
                return self.current_provider
            self.current_provider = None
        for provider in self.providers:
            last_fail_time = self.last_failure.get(provider.__name__, 0)
            if current_time - last_fail_time > self.retry_delay:
                self.current_provider = provider
                return provider
        # raise Exception("All providers are temporarily unavailable")

    def mark_failed(self, provider):
        self.last_failure[provider.__name__] = time.time()
        self.current_provider = None

auto_provider = AutoProvider(ACTIVE_PROVIDERS)

### Model Cache Functions
def extract_models_from_providers():
    """Extract unique models from all active providers."""
    all_models = set()
    for provider in ACTIVE_PROVIDERS:
        try:
            if hasattr(provider, 'models'):
                provider_models = provider.models
            elif hasattr(provider, 'get_models'):
                provider_models = provider.get_models()
            else:
                provider_models = ["gpt-4", "gpt-3.5-turbo", "gpt-4o"]
            all_models.update(provider_models)
        except Exception as e:
            logger.warning(f"Failed to extract models from {provider.__name__}: {str(e)}")
    return list(all_models)

def is_model_active(model):
    """Test if a model is active. Returns (is_active, response_time)."""
    messages = [{"role": "user", "content": "Hello"}]
    retries = len(ACTIVE_PROVIDERS)
    for attempt in range(retries):
        provider = auto_provider.get_provider()
        start_time = time.time()
        try:
            response = ChatCompletion.create(
                model=model,
                messages=messages,
                stream=False,
                provider=provider,
                timeout=10
            )
            content = getattr(response, 'content', str(response))
            if content.strip():
                response_time = time.time() - start_time
                return True, response_time
        except Exception as e:
            logger.debug(f"Model {model} failed with provider {provider.__name__}: {str(e)}")
            auto_provider.mark_failed(provider)
            continue
    return False, None

def update_active_models_cache():
    """Update model states and response times in the background."""
    def background_check():
        for model in MODELS_TO_CHECK:
            model["state"] = "checking"
        for model in MODELS_TO_CHECK:
            active, response_time = is_model_active(model["name"])
            model["state"] = "enable" if active else "disable"
            model["rate"] = response_time if active else None
    threading.Thread(target=background_check).start()

def schedule_cache_update():
    """Schedule cache updates every 1 hour."""
    update_active_models_cache()
    threading.Timer(3600, schedule_cache_update).start()

def initialize_cache():
    """Initialize the model cache with all models set to 'checking'."""
    global MODELS_TO_CHECK
    extracted_models = extract_models_from_providers()
    MODELS_TO_CHECK = [{"name": model, "state": "checking", "rate": None} for model in extracted_models]
    schedule_cache_update()

# Initialize cache at startup
initialize_cache()

def get_active_model(exclude=None):
    """Get an active model, excluding specified models."""
    if exclude is None:
        exclude = []
    for model in MODELS_TO_CHECK:
        if model["state"] == "enable" and model["name"] not in exclude:
            return model["name"]
    return None

### Telegram Bot Setup
API_TOKEN = '1607789975:AAEInQBAiHoAULJ9j7n6mBfWSssJwJV0vBY'
bot = telebot.TeleBot(API_TOKEN)

# Persian greeting messages with emojis
PERSIAN_GREETINGS = [
    "سلام {name} عزیز! 🌟 به ربات قیمت‌گذاری فولاد خوش آمدید. 🏗️",
    "درود {name}! 🚀 آماده‌ام تا در مورد قیمت‌های فولاد به شما کمک کنم. 🛠️",
    "خوش آمدید {name}! 🎉 چگونه می‌توانم در مورد فولاد به شما کمک کنم؟ 📦",
    "سلام {name}! 😊 خوشحالم که اینجا هستید. بیایید در مورد قیمت‌های فولاد صحبت کنیم. 💵",
    "سلام {name}! 🌟 به ربات اطلاعات فولاد خوش آمدید. 🔩",
    "درود {name}! ⚡ آماده‌ام تا به سوالات شما در مورد فولاد پاسخ دهم. 🏭",
    "خوش آمدید {name}! ✨ هر سوالی در مورد فولاد دارید بپرسید. 📋",
    "سلام {name}! 🤝 دستیار قیمت‌گذاری فولاد شما اینجاست. 🛠️",
    "درود {name}! 📈 بیایید با هم به بررسی قیمت‌های فولاد بپردازیم. 📈",
    "سلام {name}! 🏆 به دنیای فولاد خوش آمدید. 🏗️",
    "خوش آمدید {name}! 💡 کنجکاو در مورد فولاد هستید؟ من پاسخ‌ها را دارم. 💬",
    "سلام {name}! 🔧 آماده‌ام تا به سوالات شما در مورد فولاد پاسخ دهم. 🔧",
    "درود {name}! 🌟 چگونه می‌توانم امروز در مورد قیمت‌های فولاد به شما کمک کنم؟ 📦",
    "سلام {name}! ⚙️ ربات کارشناس فولاد شما آنلاین است. 🏭",
    "خوش آمدید {name}! 🚀 با من شروع به کاوش در قیمت‌های فولاد کنید. 🛠️"
]

def get_categories():
    """Get list of categories from sheets folder, excluding main.csv."""
    sheets_dir = 'sheets'
    files = os.listdir(sheets_dir)
    categories = [f.replace('.csv', '') for f in files if f.endswith('.csv') and f != 'main.csv']
    return categories

def get_category_data(category):
    """Fetch data for a specific category."""
    file_path = f'sheets/{category}.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

def get_main_data():
    """Fetch data from main.csv."""
    file_path = 'sheets/main.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

### Streaming GPT Response
def stream_gpt_response(chat_id, prompt, relevant_category=None):
    """Stream GPT response by editing a Telegram message with cancel option."""
    main_df = get_main_data()
    categories = get_categories()
    
    # Prepare main.csv data with emojis
    main_info = "\n".join(
        [f"📌 {row['دسته بندی']}: {row['جزئیات محصول ( دسته بندی )']} - 💰 قیمت پایه: {row['قیمت پایه محصول']}" 
         for _, row in main_df.iterrows()]
    ) if not main_df.empty else "⚠️ داده‌ای در main.csv موجود نیست."

    # Prepare category-specific data if relevant with emojis
    if relevant_category and relevant_category in categories:
        category_df = get_category_data(relevant_category)
        category_info = "\n".join(
            [f"🔹 {row.to_dict()}" for _, row in category_df.iterrows()]
        ) if not category_df.empty else f"⚠️ داده‌ای برای دسته‌بندی {relevant_category} موجود نیست."
    else:
        category_info = "🔍 هیچ دسته‌بندی خاصی در پیام شما یافت نشد."
    categories_show = "\n".join(categories)

    # Professional prompt with emoji suggestion
    full_prompt = f"""
    # نقش: دستیار بررسی قیمت آهن آلات  
    # نام: SteelBot  
    ## 🏗️ شروع مکالمه (ساده و عملی)  
    "سلام فولادی! 🔩  
    قیمت محصولات فولادی نیاز داری؟ 🎯  
    لیست قیمت‌ها مستقیماً از دیتابیس ما استخراج میشه! 📊"  
    # زبان: فارسی  
    # نکته: پاسخ‌ها باید دقیقاً مطابق داده‌های شیت و با فرمت صحیح ارائه شوند.  
    # پیشنهاد: در پاسخ‌ها از ایموجی‌هایی مثل 📌، 💰، 🔹، ⚠️، 🎯، 📊 استفاده کنید تا جذاب‌تر شوند.

    کاربر پرسیده: {prompt}
    
    داده‌های اصلی (محصولات پرطرفدار):  
    {main_info}

    تمام دسته‌بندی موجود :
    {categories_show}
    
    داده‌های دسته‌بندی مرتبط:  
    {category_info}
    
    لطفاً بر اساس این داده‌ها، به فارسی ساده و مختصر پاسخ دهید و از ایموجی‌ها استفاده کنید.
    """

    # Send initial message and set cancel flag
    sent_message = bot.send_message(chat_id, "⏳ در حال پردازش...")
    cancel_flags[sent_message.message_id] = False
    attempted_models = []

    while True:
        model = get_active_model(exclude=attempted_models)
        if not model:
            bot.edit_message_text("⚠️ متاسفانه در حال حاضر هیچ مدلی در دسترس نیست.", chat_id, sent_message.message_id)
            return
        attempted_models.append(model)

        bot.edit_message_text(f"⏳ در حال پردازش با مدل {model}...", chat_id, sent_message.message_id)

        messages = [{"role": "user", "content": full_prompt}]
        
        retries = len(ACTIVE_PROVIDERS)
        for attempt in range(retries):
            provider = auto_provider.get_provider()
            try:
                response = ChatCompletion.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    provider=provider,
                    timeout=30
                )

                accumulated_response = ""
                last_edit_time = time.time()

                try:
                    for chunk in response:
                        if cancel_flags.get(sent_message.message_id, False):
                            bot.edit_message_text("❌ عملیات لغو شد.", chat_id, sent_message.message_id)
                            return
                        content = getattr(chunk, 'content', str(chunk))
                        accumulated_response += content
                        if time.time() - last_edit_time > 0.2:
                            markup = InlineKeyboardMarkup()
                            markup.add(InlineKeyboardButton("❌ لغو عملیات ❌", callback_data='cancel'))
                            bot.edit_message_text(accumulated_response, chat_id, sent_message.message_id, reply_markup=markup)
                            last_edit_time = time.time()
                    
                    # Final response text
                    final_text = accumulated_response

                    # Prepare inline keyboard with emojis
                    markup = InlineKeyboardMarkup()
                    if relevant_category:
                        markup.add(InlineKeyboardButton(f"📦 محصولات {relevant_category.replace('_', ' ')}", 
                                                       switch_inline_query_current_chat=f"/category_{relevant_category}"))
                    else:
                        # Suggest some categories with emojis
                        suggested_categories = random.sample(categories, min(3, len(categories)))
                        for cat in suggested_categories:
                            markup.add(InlineKeyboardButton(f"🔹 دسته‌بندی {cat.replace('_', ' ')}", 
                                                           switch_inline_query_current_chat=f"/category_{cat}"))
                    
                    markup.add(InlineKeyboardButton("📋 لیست همه دسته‌بندی‌ها", switch_inline_query_current_chat="/categories"))
                    
                    bot.edit_message_text(final_text, chat_id, sent_message.message_id, reply_markup=markup)
                    return
                finally:
                    if sent_message.message_id in cancel_flags:
                        del cancel_flags[sent_message.message_id]

            except Exception as e:
                logger.warning(f"Provider {provider.__name__} failed for model {model}: {str(e)}")
                auto_provider.mark_failed(provider)
                if attempt == retries - 1:
                    break
                time.sleep(2 ** attempt)

        bot.edit_message_text(f"⚠️ خطا در پردازش با مدل {model}. در حال تلاش با مدل دیگر...", chat_id, sent_message.message_id)
        time.sleep(1)

### Bot Handlers
@bot.message_handler(commands=['start'])
def send_welcome(message):
    name = message.from_user.first_name
    welcome_text = random.choice(PERSIAN_GREETINGS).format(name=name)
    
    markup = InlineKeyboardMarkup()
    markup.row(
        InlineKeyboardButton("📦 محصولات", switch_inline_query_current_chat='/products'),
        InlineKeyboardButton("🏷️ دسته‌بندی", switch_inline_query_current_chat='/categories')
    )
    markup.add(InlineKeyboardButton("❓ راهنما", callback_data='help'))
    
    bot.send_message(message.chat.id, welcome_text, reply_markup=markup)

@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    if call.data == 'help':
        help_text = "📘 راهنمای استفاده:\n- '📦 محصولات' برای دیدن محصولات پرطرفدار\n- '🏷️ دسته‌بندی' برای لیست دسته‌ها\n- سوالات خود را مستقیم بپرسید 📝"
        bot.answer_callback_query(call.id, help_text, show_alert=True)
    
    elif call.data == 'cancel':
        message_id = call.message.message_id
        if message_id in cancel_flags:
            cancel_flags[message_id] = True
            bot.answer_callback_query(call.id, "❌ عملیات در حال لغو است...")
        else:
            bot.answer_callback_query(call.id, "✅ عملیات قبلا به پایان رسیده است.")

@bot.inline_handler(lambda query: query.query == '/products')
def show_products(inline_query):
    df = get_main_data()
    if df.empty:
        results = [InlineQueryResultArticle(
            id=str(uuid.uuid4()),
            title="⚠️ خطا",
            input_message_content=InputTextMessageContent("🏷️ هیچ محصولی در main.csv یافت نشد یا خالی است.")
        )]
    else:
        df_limited = df.head(50)  # Limit to 50 results
        results = []
        for index, row in df_limited.iterrows():
            content = (
                f"🔸 کد محصول: {row['کد محصول']}\n"
                f"🏷️ دسته‌بندی: {row['دسته بندی']}\n"
                f"📋 جزئیات: {row['جزئیات محصول ( دسته بندی )']}\n"
                f"💰 قیمت پایه: {row['قیمت پایه محصول']}\n"
                f"📦 موجودی: {row['موجودی محصول']}"
            )
            unique_id = str(uuid.uuid4())
            results.append(
                InlineQueryResultArticle(
                    id=unique_id,
                    title=f"{row['جزئیات محصول ( دسته بندی )']}",
                    description=f"💰 {row['قیمت پایه محصول']} - 📦 {row['موجودی محصول']}",
                    input_message_content=InputTextMessageContent(content)
                )
            )
    bot.answer_inline_query(inline_query.id, results)

@bot.inline_handler(lambda query: query.query == '/categories')
def show_categories(inline_query):
    categories = get_categories()
    results = []
    for cat in categories:
        unique_id = str(uuid.uuid4())
        results.append(
            InlineQueryResultArticle(
                id=unique_id,
                title=cat.replace('_', ' '),
                description=f"📦 مشاهده محصولات دسته‌بندی {cat.replace('_', ' ')}",
                input_message_content=InputTextMessageContent(f"🏷️ دسته‌بندی: {cat.replace('_', ' ')}"),
                reply_markup=InlineKeyboardMarkup().add(
                    InlineKeyboardButton("👀 مشاهده محصولات", switch_inline_query_current_chat=f"/category_{cat}")
                )
            )
        )
    bot.answer_inline_query(inline_query.id, results)

@bot.inline_handler(lambda query: query.query.startswith('/category_'))
def show_category_products(inline_query):
    category = inline_query.query.replace('/category_', '')
    df = get_category_data(category)
    if df.empty:
        results = [InlineQueryResultArticle(
            id=str(uuid.uuid4()),
            title="⚠️ خطا",
            input_message_content=InputTextMessageContent(f"🏷️ دسته‌بندی {category.replace('_', ' ')} یافت نشد یا خالی است.")
        )]
    else:
        df_limited = df.head(50)  # Limit to 50 results
        results = []
        for index, row in df_limited.iterrows():
            content = "\n".join([f"🔸 {col}: {val}" for col, val in row.to_dict().items()])
            unique_id = str(uuid.uuid4())
            results.append(
                InlineQueryResultArticle(
                    id=unique_id,
                    title=row.get('نام', f"محصول {index + 1}"),  # Adjust based on actual column names
                    description=f"📋 جزئیات محصول از {category.replace('_', ' ')}",
                    input_message_content=InputTextMessageContent(content)
                )
            )
    bot.answer_inline_query(inline_query.id, results)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    categories = get_categories()
    # Check if message contains any category name
    relevant_category = next((cat for cat in categories if cat.replace('_', ' ').strip() in message.text), None)
    stream_gpt_response(message.chat.id, message.text, relevant_category)

### Main Loop
if __name__ == '__main__':
    print("Bot is running... 🚀")
    bot.infinity_polling()