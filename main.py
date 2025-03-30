import telebot
import pandas as pd
import uuid
import random
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
        raise Exception("All providers are temporarily unavailable")

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

# Persian greeting messages
PERSIAN_GREETINGS = [
    "سلام {name} عزیز! به ربات قیمت‌گذاری فولاد خوش آمدید. 🏗️",
    "درود {name}! آماده‌ام تا در مورد قیمت‌های فولاد به شما کمک کنم. 🛠️",
    "خوش آمدید {name}! چگونه می‌توانم در مورد فولاد به شما کمک کنم؟ 📦",
    "سلام {name}! خوشحالم که اینجا هستید. بیایید در مورد قیمت‌های فولاد صحبت کنیم. 💵",
    "سلام {name}! به ربات اطلاعات فولاد خوش آمدید. 🔩",
    "درود {name}! آماده‌ام تا به سوالات شما در مورد فولاد پاسخ دهم. 🏭",
    "خوش آمدید {name}! هر سوالی در مورد فولاد دارید بپرسید. 📋",
    "سلام {name}! دستیار قیمت‌گذاری فولاد شما اینجاست. 🛠️",
    "درود {name}! بیایید با هم به بررسی قیمت‌های فولاد بپردازیم. 📈",
    "سلام {name}! به دنیای فولاد خوش آمدید. 🏗️",
    "خوش آمدید {name}! کنجکاو در مورد فولاد هستید؟ من پاسخ‌ها را دارم. 💬",
    "سلام {name}! آماده‌ام تا به سوالات شما در مورد فولاد پاسخ دهم. 🔧",
    "درود {name}! چگونه می‌توانم امروز در مورد قیمت‌های فولاد به شما کمک کنم؟ 📦",
    "سلام {name}! ربات کارشناس فولاد شما آنلاین است. 🏭",
    "خوش آمدید {name}! با من شروع به کاوش در قیمت‌های فولاد کنید. 🛠️"
]

def get_steel_data():
    """Fetch steel data from CSV."""
    df = pd.read_csv('main.csv')
    return df

### Helper function to extract suggested steels from LLM response
def get_suggested_steels(response_text, df):
    """Extract steel names mentioned in the LLM's response."""
    steel_names = df['name'].tolist()
    suggested = [name for name in steel_names if name in response_text]
    return suggested

### Streaming GPT Response
def stream_gpt_response(chat_id, prompt, found_types=None):
    """Stream GPT response by editing a Telegram message with cancel option."""
    df = get_steel_data()
    # Prepare CSV data for the prompt, avoiding redundant type in name
    if found_types:
        steel_data = df[df['types'].isin(found_types)]
        steel_info = "\n".join(
            [f"- {row['name']}{'' if row['types'] in row['name'] else f''' ({row['types']})'''}: Price {row['price']}" 
             for _, row in steel_data.iterrows()]
        )
    else:
        steel_info = "هیچ نوع فولادی در پیام شما یافت نشد."

    # Professional prompt with CSV data
    full_prompt = f"""
    # نقش: دستیار بررسی قیمت آهن آلات  
    # نام: SteelBot  
    ## 🏗️ شروع مکالمه (ساده و عملی)  
    "سلام فولادی! 🔩  
    قیمت **میلگرد، ورق یا پروفیل** نیاز داری؟  
    لیست قیمت‌ها مستقیماً از دیتابیس ما استخراج میشه!  

    🔥 فولادهای پرمتقاضی امروز:  
    • میلگرد CK45 (رنج قیمت: ۴۲۸,۰۰۰-۴۷۵,۰۰۰ ریال)  
    • تیرآهن IPE 18  
    • ورق ST37 10 میل"  
    # زبان: فارسی  
    # نکته: پاسخ‌ها باید دقیقاً مطابق داده‌های شیت و با فرمت صحیح ارائه شوند.  
    ## ❓ سوالات متداول (FAQ)  
    💬 **چطور نام فولاد رو دقیق وارد کنم؟**  
    ✅ نام رو دقیقاً مطابق لیست بنویسید:  
    مثال صحیح: "میلگرد CK45 میلگرد"  
    مثال نادرست: "میلگرد ck45"  

    💬 **آیا قیمت‌ها به روز هستند؟**  
    ✅ داده‌ها هر ۱ ساعت از شیت گوگل آپدیت می‌شوند.  

    💬 **فولاد مدنظرم تو لیست نیست!**  
    ✅ نام دقیق فولاد رو برام بنویس تا به تیم فنی اطلاع بدم.  

    کاربر پرسیده: {prompt}
    
    در زیر، داده‌های مربوط به فولادها آمده است:  
    {steel_info}
    
    لطفاً بر اساس این داده‌ها، به فارسی ساده و مختصر پاسخ دهید.
    """

    # Send initial message and set cancel flag
    sent_message = bot.send_message(chat_id, "در حال پردازش...")
    cancel_flags[sent_message.message_id] = False
    attempted_models = []

    while True:
        model = get_active_model(exclude=attempted_models)
        if not model:
            bot.edit_message_text("متاسفانه در حال حاضر هیچ مدلی در دسترس نیست.", chat_id, sent_message.message_id)
            return
        attempted_models.append(model)

        bot.edit_message_text(f"در حال پردازش با مدل {model}...", chat_id, sent_message.message_id)

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
                        # Check if the operation was canceled
                        if cancel_flags.get(sent_message.message_id, False):
                            bot.edit_message_text("عملیات لغو شد.", chat_id, sent_message.message_id)
                            return
                        content = getattr(chunk, 'content', str(chunk))
                        accumulated_response += content
                        # Update message every 0.2 seconds with cancel button
                        if time.time() - last_edit_time > 0.2:
                            markup = InlineKeyboardMarkup()
                            markup.add(InlineKeyboardButton("❌ لغو عملیات ❌", callback_data='cancel'))
                            bot.edit_message_text(accumulated_response, chat_id, sent_message.message_id, reply_markup=markup)
                            last_edit_time = time.time()
                    
                    # Final response text
                    final_text = accumulated_response

                    # Get suggested steels from LLM response
                    suggested_steels = get_suggested_steels(final_text, df)
                    if not suggested_steels:
                        # If no steels are suggested, randomly select 6
                        suggested_steels = df['name'].sample(6).tolist()
                    
                    # Create 2x3 grid for inline buttons
                    markup = InlineKeyboardMarkup()
                    for i in range(0, len(suggested_steels), 2):
                        row = []
                        for j in range(2):
                            if i + j < len(suggested_steels):
                                steel = suggested_steels[i + j]
                                row.append(InlineKeyboardButton(steel, callback_data=f'steel_{steel}'))
                        markup.add(*row)
                    
                    # Add "لیست کامل آهن‌آلات" button
                    markup.add(InlineKeyboardButton("لیست کامل آهن‌آلات 📋", switch_inline_query_current_chat="/steels"))
                    
                    # Edit final message with buttons
                    bot.edit_message_text(final_text, chat_id, sent_message.message_id, reply_markup=markup)
                    return  # Success, exit the loop
                finally:
                    # Clean up cancel flag
                    if sent_message.message_id in cancel_flags:
                        del cancel_flags[sent_message.message_id]

            except Exception as e:
                logger.warning(f"Provider {provider.__name__} failed for model {model}: {str(e)}")
                auto_provider.mark_failed(provider)
                if attempt == retries - 1:
                    break  # All providers failed for this model
                time.sleep(2 ** attempt)

        # All providers failed, try another model
        bot.edit_message_text(f"خطا در پردازش با مدل {model}. در حال تلاش با مدل دیگر...", chat_id, sent_message.message_id)
        time.sleep(1)

### Bot Handlers
@bot.message_handler(commands=['start'])
def send_welcome(message):
    name = message.from_user.first_name
    # Randomly select a Persian greeting message and insert the user's name
    welcome_text = random.choice(PERSIAN_GREETINGS).format(name=name)
    
    markup = InlineKeyboardMarkup()
    markup.row(
        InlineKeyboardButton("میلگرد 🛠️", callback_data='type_میلگرد'),
        InlineKeyboardButton("ورق 📦", callback_data='type_ورق')
    )
    markup.row(
        InlineKeyboardButton("آخرین قیمت‌ها 📈", callback_data='latest'),
        InlineKeyboardButton("راهنما ❓", callback_data='help')
    )
    markup.add(InlineKeyboardButton("لیست کامل آهن‌آلات 📋", switch_inline_query_current_chat="/steels"))
    
    bot.send_message(message.chat.id, welcome_text, reply_markup=markup)

@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    df = get_steel_data()
    
    if call.data.startswith('type_'):
        steel_type = call.data.split('_')[1]
        results = df[df['types'] == steel_type]
        
        if not results.empty:
            results = results[:10] if len(results) >= 10 else results
            # Prepare list of steels for this type, filtering type from name if redundant
            steel_list = "\n".join(
                [f"🔧 {row['name']}{'' if row['types'] in row['name'] else f'\n (🏷️ {row['types']})'} \n- 💵 {row['price']}\n{'➖'*4}\n" 
                 for _, row in results.iterrows()]
            )
            response_text = f"لیست فولادهای نوع {steel_type}:\n{steel_list}"
            
            # Create inline keyboard with "بازگشت" button
            markup = InlineKeyboardMarkup()
            markup.add(InlineKeyboardButton("بازگشت ↩️", callback_data='back_to_start'))
            
            # Edit the current message
            bot.edit_message_text(response_text, chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=markup)
            bot.answer_callback_query(call.id)  # Acknowledge the callback
        else:
            bot.answer_callback_query(call.id, "نوع مورد نظر یافت نشد", show_alert=True)
    
    elif call.data == 'back_to_start':
        # Revert to the /start message
        name = call.from_user.first_name
        welcome_text = random.choice(PERSIAN_GREETINGS).format(name=name)
        
        markup = InlineKeyboardMarkup()
        markup.row(
            InlineKeyboardButton("میلگرد 🛠️", callback_data='type_میلگرد'),
            InlineKeyboardButton("ورق 📦", callback_data='type_ورق')
        )
        markup.row(
            InlineKeyboardButton("آخرین قیمت‌ها 📈", callback_data='latest'),
            InlineKeyboardButton("راهنما ❓", callback_data='help')
        )
        markup.add(InlineKeyboardButton("لیست کامل آهن‌آلات 📋", switch_inline_query_current_chat="/steels"))
        
        bot.edit_message_text(welcome_text, chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=markup)
        bot.answer_callback_query(call.id)
    
    elif call.data == 'latest':
        latest = df.tail(3)
        response = "آخرین قیمت‌های ثبت شده:\n\n" + "\n\n".join(
            [f"🔹 {row['name']}\n💰 {row['price']}" for _, row in latest.iterrows()]
        )
        bot.answer_callback_query(call.id, response, show_alert=True)
    
    elif call.data == 'help':
        help_text = "راهنمای استفاده:\n- از دکمه‌های زیر برای دسترسی سریع استفاده کنید\n- برای جستجو از /steels استفاده نمایید\n- سوالات خود را مستقیم بپرسید"
        bot.answer_callback_query(call.id, help_text, show_alert=True)
    
    elif call.data == 'cancel':
        message_id = call.message.message_id
        if message_id in cancel_flags:
            cancel_flags[message_id] = True
            bot.answer_callback_query(call.id, "عملیات در حال لغو است...")
        else:
            bot.answer_callback_query(call.id, "عملیات قبلا به پایان رسیده است.")

@bot.inline_handler(lambda query: query.query == '/steels')
def show_steels(inline_query):
    df = get_steel_data()
    # Limit to 50 results to comply with Telegram's API restriction
    df_limited = df.head(50)  # Take the first 50 rows
    results = []
    seen_types = set()  # Track seen types to handle duplicates
    
    for index, row in df_limited.iterrows():
        content = InputTextMessageContent(
            f"🔩 {row['name']}\n🏷️ نوع: {row['types']}\n💰 قیمت: {row['price']}"
        )
        # Split type into words and remove duplicates, then rejoin
        type_words = row['types'].split()
        unique_type = " ".join(sorted(set(type_words), key=type_words.index))
        
        # If this type was seen before, append a unique identifier
        if unique_type in seen_types:
            title = f"{unique_type} ({index + 1})"
        else:
            title = unique_type
            seen_types.add(unique_type)
            
        # Use a unique ID to avoid RESULT_ID_DUPLICATE error
        unique_id = str(uuid.uuid4())
        results.append(
            InlineQueryResultArticle(
                id=unique_id,
                title=title,  # Use filtered type as title
                description=f"{row['types']} - {row['price']}",
                input_message_content=content
            )
        )
    
    bot.answer_inline_query(inline_query.id, results)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    df = get_steel_data()
    steel_types = df['types'].unique()
    
    # Define specific steel-related keywords to check in the message
    steel_keywords = ["فولاد", "تسمه", "میلگرد", "ورق"]
    
    # Check if any steel keywords are in the user's message
    found_keywords = [keyword for keyword in steel_keywords if keyword in message.text]
    
    if found_keywords:
        # Filter types that match the found keywords
        found_types = [t for t in steel_types if any(keyword in t for keyword in found_keywords)]
    else:
        # If no keywords are found, send all types to the LLM
        found_types = list(steel_types)
    
    stream_gpt_response(message.chat.id, message.text, found_types)

### Main Loop
if __name__ == '__main__':
    print("Bot is running...")
    bot.infinity_polling()