import { MnistDcganPage } from './app.po';

describe('mnist-dcgan App', () => {
  let page: MnistDcganPage;

  beforeEach(() => {
    page = new MnistDcganPage();
  });

  it('should display welcome message', () => {
    page.navigateTo();
    expect(page.getParagraphText()).toEqual('Welcome to app!!');
  });
});
