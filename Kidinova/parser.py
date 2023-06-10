import requests
import pandas as pd

from config import access_token, version


def get_group_info(group_screen_name):
    field = 'activity,can_post,links,description,site,age_limits,wall,market,type,start_date,verified,status,contacts,counters,members_count,can_see_all_posts,photo_50,wiki_page,fixed_post,main_section'
    response = requests.get(f'https://api.vk.com/method/groups.getById',
                            params={'access_token': access_token, 'v': version,
                                    'group_id': group_screen_name, 'fields': field}
                            )

    data = response.json()
    if 'response' not in data:
        return None
    for i in range(len(data['response'])):
        if 'links' in data['response']:
            data['response'][i]['links_count'] = len(data['response'][i]['links'])
        if 'contacts' in data['response']:
            data['response'][i]['contacts_count'] = len(data['response'][i]['contacts'])
    columns = ['id', 'description', 'members_count', 'activity', 'status',
               'name', 'screen_name', 'is_closed', 'links_count', 'contacts_count',
               'age_limits', 'wall', 'wiki_page', 'main_section', 'site']
    df = pd.json_normalize(data['response'])
    for col in columns:
        if col not in df.columns:
            df[col] = [None] * df.shape[0]
    return df[columns]


def parse_group(group_id):
    response = requests.get(f'https://api.vk.com/method/wall.get',
                            params={'access_token': access_token, 'v': version,
                                    'owner_id': group_id,
                                    'count': 100}
                            )
    data = response.json()
    if 'response' not in data:
        return None
    for i in range(len(data['response']['items'])):
        if 'attachments' in data['response']['items'][i]:
            attachments = [x['type'] for x in data['response']['items'][i]['attachments']]
            data['response']['items'][i]['audio_count'] = attachments.count('audio')
            data['response']['items'][i]['photo_count'] = attachments.count('photo')
            data['response']['items'][i]['video_count'] = attachments.count('video')
            data['response']['items'][i]['doc_count'] = attachments.count('doc')
            if len(attachments) > 0 and 'photo' == attachments[0]:
                data['response']['items'][i]['photo_url'] = data['response']['items'][i]['attachments'][0]['photo']['sizes'][0]['url']
    columns = ['date', 'id', 'owner_id', 'text',
               'comments.count', 'likes.count', 'reposts.count',
               'views.count', 'photo_url', 'audio_count', 'photo_count', 'video_count', 'doc_count']
    df = pd.json_normalize(data['response']['items'])
    for col in columns:
        if col not in df.columns:
            df[col] = [None] * df.shape[0]
    return df[columns]


def parse(url):
    group_id = url.split('/')[-1]

    group_info_df = get_group_info(group_id)
    if group_info_df is None:
        print(f'Sorry, no such group {group_id} :(')
        return None, None

    group_id = (-1) * group_info_df.id.values[0]

    group_posts_df = parse_group(group_id)
    return group_info_df, group_posts_df
